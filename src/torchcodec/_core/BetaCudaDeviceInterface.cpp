// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <map>
#include <mutex>
#include <vector>
#include "StableABICompat.h"
#include "ValidationUtils.h"

#include "BetaCudaDeviceInterface.h"

#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "Logging.h"
#include "NVDECCache.h"

#include "NVCUVIDRuntimeLoader.h"
#include "color_conversion.h"
#include "nvcuvid_include/cuviddec.h"
#include "nvcuvid_include/nvcuvid.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

namespace {

// Per-device cache for cuvidGetDecoderCaps results.
// The key is a tuple of (device index, codec type, chroma format, bit depth
// minus 8).
struct DecoderCapsCache {
  using Key =
      std::tuple<int, cudaVideoCodec, cudaVideoChromaFormat, unsigned int>;
  std::map<Key, CUVIDDECODECAPS> cache;
  std::mutex mutex;

  std::pair<CUresult, CUVIDDECODECAPS> get_decoder_caps(
      int device_index,
      cudaVideoCodec codec_type,
      cudaVideoChromaFormat chroma_format,
      unsigned int bit_depth_minus8) {
    Key key{device_index, codec_type, chroma_format, bit_depth_minus8};

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return {CUDA_SUCCESS, it->second};
    }

    CUVIDDECODECAPS caps = {};
    caps.eCodecType = codec_type;
    caps.eChromaFormat = chroma_format;
    caps.nBitDepthMinus8 = bit_depth_minus8;

    CUresult result = cuvidGetDecoderCaps(&caps);
    if (result == CUDA_SUCCESS) {
      cache[key] = caps;
    }
    return {result, caps};
  }
};

static DecoderCapsCache& get_decoder_caps_cache() {
  static DecoderCapsCache cache;
  return cache;
}

cudaVideoSurfaceFormat get_preferred_surface_format(OutputDtype output_dtype) {
  return output_dtype == OutputDtype::FLOAT32 ? cudaVideoSurfaceFormat_P016
                                              : cudaVideoSurfaceFormat_NV12;
}

static bool g_cuda_nvdec = register_device_interface(
    DeviceInterfaceKey(kStableCUDA, /*variant=*/"default"),
    [](const StableDevice& device) {
      return new BetaCudaDeviceInterface(device);
    });

static int CUDAAPI
pfn_sequence_callback(void* p_user_data, CUVIDEOFORMAT* video_format) {
  auto decoder = static_cast<BetaCudaDeviceInterface*>(p_user_data);
  return decoder->stream_property_change(video_format);
}

static int CUDAAPI
pfn_decode_picture_callback(void* p_user_data, CUVIDPICPARAMS* pic_params) {
  auto decoder = static_cast<BetaCudaDeviceInterface*>(p_user_data);
  return decoder->frame_ready_for_decoding(pic_params);
}

static int CUDAAPI pfn_display_picture_callback(
    void* p_user_data,
    CUVIDPARSERDISPINFO* disp_info) {
  auto decoder = static_cast<BetaCudaDeviceInterface*>(p_user_data);
  return decoder->frame_ready_in_display_order(disp_info);
}

static UniqueCUvideodecoder create_decoder(
    CUVIDEOFORMAT* video_format,
    cudaVideoSurfaceFormat surface_format) {
  // Decoder creation parameters, most are taken from DALI
  CUVIDDECODECREATEINFO decoder_params = {};
  decoder_params.bitDepthMinus8 = video_format->bit_depth_luma_minus8;
  decoder_params.ChromaFormat = video_format->chroma_format;
  decoder_params.OutputFormat = surface_format;
  decoder_params.ulCreationFlags = cudaVideoCreate_Default;
  decoder_params.CodecType = video_format->codec;
  decoder_params.ulHeight = video_format->coded_height;
  decoder_params.ulWidth = video_format->coded_width;
  decoder_params.ulMaxHeight = video_format->coded_height;
  decoder_params.ulMaxWidth = video_format->coded_width;
  decoder_params.ulTargetHeight =
      video_format->display_area.bottom - video_format->display_area.top;
  decoder_params.ulTargetWidth =
      video_format->display_area.right - video_format->display_area.left;
  decoder_params.ulNumDecodeSurfaces = video_format->min_num_decode_surfaces;
  // We should only ever need 1 output surface, since we process frames
  // sequentially, and we always unmap the previous frame before mapping a new
  // one.
  // TODONVDEC P3: set this to 2, allow for 2 frames to be mapped at a time, and
  // benchmark to see if this makes any difference.
  decoder_params.ulNumOutputSurfaces = 1;
  decoder_params.display_area.left = video_format->display_area.left;
  decoder_params.display_area.right = video_format->display_area.right;
  decoder_params.display_area.top = video_format->display_area.top;
  decoder_params.display_area.bottom = video_format->display_area.bottom;

  CUvideodecoder* decoder = new CUvideodecoder();
  CUresult result = cuvidCreateDecoder(decoder, &decoder_params);
  STD_TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create NVDEC decoder: ", result);
  return UniqueCUvideodecoder(decoder, CUvideoDecoderDeleter{});
}

std::optional<cudaVideoChromaFormat> validate_chroma_support(
    const AVPixFmtDescriptor* desc) {
  // Return the corresponding cudaVideoChromaFormat if supported, std::nullopt
  // otherwise.
  STD_TORCH_CHECK(desc != nullptr, "desc can't be null");

  if (desc->nb_components == 1) {
    return cudaVideoChromaFormat_Monochrome;
  } else if (desc->nb_components >= 3 && !(desc->flags & AV_PIX_FMT_FLAG_RGB)) {
    // Make sure it's YUV: has chroma planes and isn't RGB
    if (desc->log2_chroma_w == 0 && desc->log2_chroma_h == 0) {
      return cudaVideoChromaFormat_444; // 1x1 subsampling = 4:4:4
    } else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 1) {
      return cudaVideoChromaFormat_420; // 2x2 subsampling = 4:2:0
    } else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 0) {
      return cudaVideoChromaFormat_422; // 2x1 subsampling = 4:2:2
    }
  }

  return std::nullopt;
}

std::optional<cudaVideoCodec> validate_codec_support(AVCodecID codec_id) {
  // Return the corresponding cudaVideoCodec if supported, std::nullopt
  // otherwise
  // Note that we currently return nullopt (and thus fallback to CPU) for some
  // codecs that are technically supported by NVDEC, see comment below.
  switch (codec_id) {
    case AV_CODEC_ID_H264:
      return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC:
      return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_AV1:
      return cudaVideoCodec_AV1;
    case AV_CODEC_ID_VP9:
      return cudaVideoCodec_VP9;
    case AV_CODEC_ID_VP8:
      return cudaVideoCodec_VP8;
    case AV_CODEC_ID_MPEG4:
      return cudaVideoCodec_MPEG4;
    // Formats below are currently not tested, but they should "mostly" work.
    // MPEG1 was briefly locally tested and it was ok-ish despite duration being
    // off. Since they're far less popular, we keep them disabled by default but
    // we can consider enabling them upon user requests.
    // case AV_CODEC_ID_MPEG1VIDEO:
    //   return cudaVideoCodec_MPEG1;
    // case AV_CODEC_ID_MPEG2VIDEO:
    //   return cudaVideoCodec_MPEG2;
    // case AV_CODEC_ID_MJPEG:
    //   return cudaVideoCodec_JPEG;
    // case AV_CODEC_ID_VC1:
    //   return cudaVideoCodec_VC1;
    default:
      return std::nullopt;
  }
}

std::optional<cudaVideoSurfaceFormat> get_nvdec_surface_format(
    const StableDevice& device,
    const SharedAVCodecContext& codec_context,
    OutputDtype output_dtype) {
  // Return the surface format to use for NVDEC decoding if the stream is
  // supported, or nullopt to fall back to CPU.

  auto codec_type = validate_codec_support(codec_context->codec_id);
  if (!codec_type.has_value()) {
    return std::nullopt;
  }

  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(codec_context->pix_fmt);
  if (!desc) {
    return std::nullopt;
  }

  auto chroma_format = validate_chroma_support(desc);
  if (!chroma_format.has_value()) {
    return std::nullopt;
  }

  auto bit_depth_minus8 = static_cast<unsigned int>(desc->comp[0].depth - 8);
  auto [result, caps] = get_decoder_caps_cache().get_decoder_caps(
      get_device_index(device),
      codec_type.value(),
      chroma_format.value(),
      bit_depth_minus8);
  if (result != CUDA_SUCCESS) {
    return std::nullopt;
  }

  if (!caps.bIsSupported) {
    return std::nullopt;
  }

  auto coded_width = static_cast<unsigned int>(codec_context->coded_width);
  auto coded_height = static_cast<unsigned int>(codec_context->coded_height);
  if (coded_width < static_cast<unsigned int>(caps.nMinWidth) ||
      coded_height < static_cast<unsigned int>(caps.nMinHeight) ||
      coded_width > caps.nMaxWidth || coded_height > caps.nMaxHeight) {
    return std::nullopt;
  }

  // See nMaxMBCount in cuviddec.h
  constexpr unsigned int macroblock_constant = 256;
  if (coded_width * coded_height / macroblock_constant > caps.nMaxMBCount) {
    return std::nullopt;
  }

  auto preferred_format = get_preferred_surface_format(output_dtype);
  if ((caps.nOutputFormatMask >> preferred_format) & 1) {
    return preferred_format;
  }

  // P016 is typically not supported on 8-bit SDR content. In such cases, we
  // try to fall back to NV12 if supported:
  // NVDEC will decode to NV12, our kernel will do NV12 -> RGB producing
  // uint8, and maybePermuteAndConvertToFloat32 will cast uint8 -> float32.
  // For HDR content, NV12 would lose precision, so we fall back to CPU instead.
  if (preferred_format == cudaVideoSurfaceFormat_P016 &&
      bit_depth_minus8 == 0 &&
      ((caps.nOutputFormatMask >> cudaVideoSurfaceFormat_NV12) & 1)) {
    return cudaVideoSurfaceFormat_NV12;
  }

  return std::nullopt;
}

// Callback for freeing CUDA memory associated with AVFrame see where it's used
// for more details.
void cuda_buffer_free_callback(void* opaque, [[maybe_unused]] uint8_t* data) {
  cudaFree(opaque);
}

} // namespace

BetaCudaDeviceInterface::BetaCudaDeviceInterface(const StableDevice& device)
    : DeviceInterface(device) {
  STD_TORCH_CHECK(g_cuda_nvdec, "NvdecCudaDeviceInterface was not registered!");
  STD_TORCH_CHECK(
      device_.type() == kStableCUDA, "Unsupported device: must be CUDA");

  initialize_cuda_context_with_pytorch(device_);

  nvcuvid_available_ = load_nvcuvid_library();
}

void BetaCudaDeviceInterface::initialize_video(
    const AVStream* av_stream,
    const UniqueDecodingAVFormatContext& av_format_ctx,
    const VideoStreamOptions& video_stream_options,
    const std::vector<std::unique_ptr<Transform>>& transforms,
    const std::optional<FrameDims>& resized_output_dims) {
  STD_TORCH_CHECK(av_stream != nullptr, "AVStream cannot be null");
  rotation_ = rotation_from_degrees(get_rotation_from_stream(av_stream));
  output_dtype_ = video_stream_options.output_dtype;

  auto maybe_surface_format = nvcuvid_available_
      ? get_nvdec_surface_format(device_, codec_context_, output_dtype_)
      : std::nullopt;

  if (!maybe_surface_format.has_value()) {
    if (!nvcuvid_available_) {
      TC_LOG("NVCUVID library not available; falling back to CPU decoding.");
    } else {
      TC_LOG(
          "Video stream not supported by NVDEC; falling back to CPU decoding.");
    }
    cpu_fallback_ = create_device_interface(kStableCPU);
    STD_TORCH_CHECK(
        cpu_fallback_ != nullptr, "Failed to create CPU device interface");
    cpu_fallback_->initialize(codec_context_);
    cpu_fallback_->initialize_video(
        av_stream,
        av_format_ctx,
        video_stream_options,
        transforms,
        resized_output_dims);
    return;
  }

  surface_format_ = maybe_surface_format.value();
  time_base_ = av_stream->time_base;
  frame_rate_avg_from_ffmpeg_ = av_stream->r_frame_rate;

  const AVCodecParameters* codec_par = av_stream->codecpar;
  STD_TORCH_CHECK(codec_par != nullptr, "CodecParameters cannot be null");

  initialize_bsf(codec_par, av_format_ctx);

  // Create parser. Default values that aren't obvious are taken from DALI.
  CUVIDPARSERPARAMS parser_params = {};
  auto codec_type = validate_codec_support(codec_par->codec_id);
  STD_TORCH_CHECK(
      codec_type.has_value(),
      "This should never happen, we should be using the CPU fallback by now. "
      "Please report a bug.");
  parser_params.CodecType = codec_type.value();
  parser_params.ulMaxNumDecodeSurfaces = 8;
  parser_params.ulMaxDisplayDelay = 0;
  // Callback setup, all are triggered by the parser within a call
  // to cuvidParseVideoData
  parser_params.pUserData = this;
  parser_params.pfnSequenceCallback = pfn_sequence_callback;
  parser_params.pfnDecodePicture = pfn_decode_picture_callback;
  parser_params.pfnDisplayPicture = pfn_display_picture_callback;

  // Some containers (e.g. MP4/MOV) store codec config (H.264 SPS/PPS,
  // MPEG-4 VOS/VOL, etc.) in extradata rather than inline in the
  // bitstream. The NVCUVID parser needs this data to initialize, so we
  // pass it via pExtVideoInfo. Same approach as DALI and FFmpeg cuviddec.
  // DALI does the same thing
  // https://github.com/NVIDIA/DALI/blob/ae79f316ae9b14c464d9cb98465f7f783da9ea89/dali/operators/video/frames_decoder_gpu.cc#L402-L408
  if (codec_par->extradata_size > 0) {
    auto seqhdr_size = std::min(
        static_cast<size_t>(codec_par->extradata_size),
        sizeof(parser_ext_info_.raw_seqhdr_data));
    parser_ext_info_.format.seqhdr_data_length = seqhdr_size;
    memcpy(parser_ext_info_.raw_seqhdr_data, codec_par->extradata, seqhdr_size);
    parser_params.pExtVideoInfo = &parser_ext_info_;
  }

  CUresult result = cuvidCreateVideoParser(&video_parser_, &parser_params);
  STD_TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create video parser: ", result);
}

BetaCudaDeviceInterface::~BetaCudaDeviceInterface() {
  if (decoder_) {
    // DALI doesn't seem to do any particular cleanup of the decoder before
    // sending it to the cache, so we probably don't need to do anything either.
    // Just to be safe, we flush.
    // What happens to those decode surfaces that haven't yet been mapped is
    // unclear.
    flush();
    unmap_previous_frame();
    NVDECCache::get_cache(device_).return_decoder(
        &video_format_, surface_format_, std::move(decoder_));
  }

  if (video_parser_) {
    cuvidDestroyVideoParser(video_parser_);
    video_parser_ = nullptr;
  }
}

void BetaCudaDeviceInterface::initialize(
    const SharedAVCodecContext& codec_context) {
  codec_context_ = codec_context;
}

void BetaCudaDeviceInterface::initialize_bsf(
    const AVCodecParameters* codec_par,
    const UniqueDecodingAVFormatContext& av_format_ctx) {
  // Setup bit stream filters (BSF):
  // https://ffmpeg.org/doxygen/7.0/group__lavc__bsf.html
  // This is only needed for some formats, like H264 or HEVC.

  STD_TORCH_CHECK(codec_par != nullptr, "codecPar cannot be null");
  STD_TORCH_CHECK(av_format_ctx != nullptr, "AVFormatContext cannot be null");
  STD_TORCH_CHECK(
      av_format_ctx->iformat != nullptr,
      "AVFormatContext->iformat cannot be null");
  std::string filter_name;

  // Matching logic is taken from DALI
  switch (codec_par->codec_id) {
    case AV_CODEC_ID_H264: {
      const std::string format_name = av_format_ctx->iformat->long_name
          ? av_format_ctx->iformat->long_name
          : "";

      if (format_name == "QuickTime / MOV" ||
          format_name == "FLV (Flash Video)" ||
          format_name == "Matroska / WebM" ||
          format_name == "raw H.264 video") {
        filter_name = "h264_mp4toannexb";
      }
      break;
    }

    case AV_CODEC_ID_HEVC: {
      const std::string format_name = av_format_ctx->iformat->long_name
          ? av_format_ctx->iformat->long_name
          : "";

      if (format_name == "QuickTime / MOV" ||
          format_name == "FLV (Flash Video)" ||
          format_name == "Matroska / WebM" || format_name == "raw HEVC video") {
        filter_name = "hevc_mp4toannexb";
      }
      break;
    }
    case AV_CODEC_ID_MPEG4: {
      const std::string format_name =
          av_format_ctx->iformat->name ? av_format_ctx->iformat->name : "";
      if (format_name == "avi") {
        filter_name = "mpeg4_unpack_bframes";
      }
      break;
    }

    default:
      // No bitstream filter needed for other codecs
      break;
  }

  if (filter_name.empty()) {
    // Only initialize BSF if we actually need one
    return;
  }

  const AVBitStreamFilter* av_bsf = av_bsf_get_by_name(filter_name.c_str());
  STD_TORCH_CHECK(
      av_bsf != nullptr, "Failed to find bitstream filter: ", filter_name);

  AVBSFContext* av_bsf_context = nullptr;
  int ret_val = av_bsf_alloc(av_bsf, &av_bsf_context);
  STD_TORCH_CHECK(
      ret_val >= AVSUCCESS,
      "Failed to allocate bitstream filter: ",
      get_ffmpeg_error_string_from_error_code(ret_val));

  bitstream_filter_.reset(av_bsf_context);

  ret_val = avcodec_parameters_copy(bitstream_filter_->par_in, codec_par);
  STD_TORCH_CHECK(
      ret_val >= AVSUCCESS,
      "Failed to copy codec parameters: ",
      get_ffmpeg_error_string_from_error_code(ret_val));

  ret_val = av_bsf_init(bitstream_filter_.get());
  STD_TORCH_CHECK(
      ret_val == AVSUCCESS,
      "Failed to initialize bitstream filter: ",
      get_ffmpeg_error_string_from_error_code(ret_val));
}

// This callback is called by the parser within cuvidParseVideoData when there
// is a change in the stream's properties (like resolution change), as specified
// by CUVIDEOFORMAT. Particularly (but not just!), this is called at the very
// start of the stream.
// TODONVDEC P1: Code below mostly assume this is called only once at the start,
// we should handle the case of multiple calls. Probably need to flush buffers,
// etc.
int BetaCudaDeviceInterface::stream_property_change(
    CUVIDEOFORMAT* video_format) {
  STD_TORCH_CHECK(video_format != nullptr, "Invalid video format");

  video_format_ = *video_format;

  if (video_format_.min_num_decode_surfaces == 0) {
    // Same as DALI's fallback
    video_format_.min_num_decode_surfaces = 20;
  }

  if (!decoder_) {
    decoder_ = NVDECCache::get_cache(device_).get_decoder(
        video_format, surface_format_);

    if (!decoder_) {
      // TODONVDEC P2: consider re-configuring an existing decoder instead of
      // re-creating one. See docs, see DALI. Re-configuration doesn't seem to
      // be enabled in DALI by default.
      decoder_ = create_decoder(video_format, surface_format_);
    }

    STD_TORCH_CHECK(decoder_, "Failed to get or create decoder");
  }

  // DALI also returns min_num_decode_surfaces from this function. This
  // instructs the parser to reset its ulMaxNumDecodeSurfaces field to this
  // value.
  return static_cast<int>(video_format_.min_num_decode_surfaces);
}

// Moral equivalent of avcodec_send_packet(). Here, we pass the AVPacket down to
// the NVCUVID parser.
int BetaCudaDeviceInterface::send_packet(ReferenceAVPacket& packet) {
  if (cpu_fallback_) {
    return cpu_fallback_->send_packet(packet);
  }

  STD_TORCH_CHECK(
      packet.get() && packet->data && packet->size > 0,
      "sendPacket received an empty packet, this is unexpected, please report.");

  // Apply BSF if needed. We want applyBSF to return a *new* filtered packet, or
  // the original one if no BSF is needed. This new filtered packet must be
  // allocated outside of applyBSF: if it were allocated inside applyBSF, it
  // would be destroyed at the end of the function, leaving us with a dangling
  // reference.
  AutoAVPacket filtered_auto_packet;
  ReferenceAVPacket filtered_packet(filtered_auto_packet);
  ReferenceAVPacket& packet_to_send = apply_bsf(packet, filtered_packet);

  CUVIDSOURCEDATAPACKET cuvid_packet = {};
  cuvid_packet.payload = packet_to_send->data;
  cuvid_packet.payload_size = packet_to_send->size;
  cuvid_packet.flags = CUVID_PKT_TIMESTAMP;
  cuvid_packet.timestamp = packet_to_send->pts;

  return send_cuvid_packet(cuvid_packet);
}

int BetaCudaDeviceInterface::send_eof_packet() {
  if (cpu_fallback_) {
    return cpu_fallback_->send_eof_packet();
  }

  CUVIDSOURCEDATAPACKET cuvid_packet = {};
  cuvid_packet.flags = CUVID_PKT_ENDOFSTREAM;
  eof_sent_ = true;

  return send_cuvid_packet(cuvid_packet);
}

int BetaCudaDeviceInterface::send_cuvid_packet(
    CUVIDSOURCEDATAPACKET& cuvid_packet) {
  CUresult result = cuvidParseVideoData(video_parser_, &cuvid_packet);
  return result == CUDA_SUCCESS ? AVSUCCESS : AVERROR_EXTERNAL;
}

ReferenceAVPacket& BetaCudaDeviceInterface::apply_bsf(
    ReferenceAVPacket& packet,
    ReferenceAVPacket& filtered_packet) {
  if (!bitstream_filter_) {
    return packet;
  }

  int ret_val = av_bsf_send_packet(bitstream_filter_.get(), packet.get());
  STD_TORCH_CHECK(
      ret_val >= AVSUCCESS,
      "Failed to send packet to bitstream filter: ",
      get_ffmpeg_error_string_from_error_code(ret_val));

  // TODO P1: the docs mention there can theoretically be multiple output
  // packets for a single input, i.e. we may need to call av_bsf_receive_packet
  // more than once. We should figure out whether that applies to the BSF we're
  // using.
  ret_val =
      av_bsf_receive_packet(bitstream_filter_.get(), filtered_packet.get());
  STD_TORCH_CHECK(
      ret_val >= AVSUCCESS,
      "Failed to receive packet from bitstream filter: ",
      get_ffmpeg_error_string_from_error_code(ret_val));

  return filtered_packet;
}

// Parser triggers this callback within cuvidParseVideoData when a frame is
// ready to be decoded, i.e. the parser received all the necessary packets for a
// given frame. It means we can send that frame to be decoded by the hardware
// NVDEC decoder by calling cuvidDecodePicture.
int BetaCudaDeviceInterface::frame_ready_for_decoding(
    CUVIDPICPARAMS* pic_params) {
  STD_TORCH_CHECK(pic_params != nullptr, "Invalid picture parameters");
  STD_TORCH_CHECK(decoder_, "Decoder not initialized before picture decode");
  // Send frame to be decoded by NVDEC. This may or may not block, depending on
  // the internal state of the NVDEC. Presumably, when it blocks, it gets
  // automatically unblocked once a frame has been decoded, although how and
  // when it happens is unclear. The docs say:
  // > cuvidDecodePicture() will stall if wait queue on NVDEC inside driver is
  //   full.
  // and cuviddec.h says:
  // > cuvidDecodePicture may block the calling thread if there are too many
  //   pictures pending in the decode queue.
  CUresult result = cuvidDecodePicture(*decoder_.get(), pic_params);

  // Yes, you're reading that right, 0 means error, 1 means success
  return (result == CUDA_SUCCESS);
}

int BetaCudaDeviceInterface::frame_ready_in_display_order(
    CUVIDPARSERDISPINFO* disp_info) {
  ready_frames_.push(*disp_info);
  return 1; // success
}

// Moral equivalent of avcodec_receive_frame().
int BetaCudaDeviceInterface::receive_frame(UniqueAVFrame& av_frame) {
  if (cpu_fallback_) {
    return cpu_fallback_->receive_frame(av_frame);
  }

  if (ready_frames_.empty()) {
    // No frame found, instruct caller to try again later after sending more
    // packets, or to stop if EOF was already sent.
    return eof_sent_ ? AVERROR_EOF : AVERROR(EAGAIN);
  }

  CUVIDPARSERDISPINFO disp_info = ready_frames_.front();
  ready_frames_.pop();

  CUVIDPROCPARAMS proc_params = {};
  proc_params.progressive_frame = disp_info.progressive_frame;
  proc_params.top_field_first = disp_info.top_field_first;
  proc_params.unpaired_field = disp_info.repeat_first_field < 0;
  // We set the NVDEC stream to the current stream. It will be waited upon
  // by the color conversion stream before any color conversion.
  // Re types: we get a cudaStream_t from PyTorch but it's interchangeable with
  // CUstream
  proc_params.output_stream =
      reinterpret_cast<CUstream>(get_current_cuda_stream(device_.index()));

  CUdeviceptr frame_ptr = 0;
  unsigned int pitch = 0;

  // We know the frame we want was sent to the hardware decoder, but now we need
  // to "map" it to an "output surface" before we can use its data. This is a
  // blocking calls that waits until the frame is fully decoded and ready to be
  // used.
  // When a frame is mapped to an output surface, it needs to be unmapped
  // eventually, so that the decoder can re-use the output surface. Failing to
  // unmap will cause map to eventually fail. DALI unmaps frames almost
  // immediately  after mapping them: they do the color-conversion in-between,
  // which involves a copy of the data, so that works.
  // We, OTOH, will do the color-conversion later, outside of ReceiveFrame(). So
  // we unmap here: just before mapping a new frame. At that point we know that
  // the previously-mapped frame is no longer needed: it was either
  // color-converted (with a copy), or that's a frame that was discarded in
  // SingleStreamDecoder. Either way, the underlying output surface can be
  // safely re-used.
  unmap_previous_frame();
  CUresult result = cuvidMapVideoFrame(
      *decoder_.get(),
      disp_info.picture_index,
      &frame_ptr,
      &pitch,
      &proc_params);
  if (result != CUDA_SUCCESS) {
    return AVERROR_EXTERNAL;
  }
  previously_mapped_frame_ = frame_ptr;

  av_frame = convert_cuda_frame_to_av_frame(frame_ptr, pitch, disp_info);

  return AVSUCCESS;
}

void BetaCudaDeviceInterface::unmap_previous_frame() {
  if (previously_mapped_frame_ == 0) {
    return;
  }
  CUresult result =
      cuvidUnmapVideoFrame(*decoder_.get(), previously_mapped_frame_);
  STD_TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to unmap previous frame: ", result);
  previously_mapped_frame_ = 0;
}

UniqueAVFrame BetaCudaDeviceInterface::convert_cuda_frame_to_av_frame(
    CUdeviceptr frame_ptr,
    unsigned int pitch,
    const CUVIDPARSERDISPINFO& disp_info) {
  STD_TORCH_CHECK(frame_ptr != 0, "Invalid CUDA frame pointer");

  // Get frame dimensions from video format display area (not coded dimensions)
  // This matches DALI's approach and avoids padding issues
  int width =
      video_format_.display_area.right - video_format_.display_area.left;
  int height =
      video_format_.display_area.bottom - video_format_.display_area.top;

  STD_TORCH_CHECK(width > 0 && height > 0, "Invalid frame dimensions");
  STD_TORCH_CHECK(
      pitch >= static_cast<unsigned int>(width), "Pitch must be >= width");

  UniqueAVFrame av_frame(av_frame_alloc());
  STD_TORCH_CHECK(av_frame.get() != nullptr, "Failed to allocate AVFrame");

  av_frame->width = width;
  av_frame->height = height;
  av_frame->format = (surface_format_ == cudaVideoSurfaceFormat_P016)
      ? AV_PIX_FMT_P016LE
      : AV_PIX_FMT_NV12;
  av_frame->pts = disp_info.timestamp;

  // TODONVDEC P2: We compute the duration based on average frame rate info, so
  // so if the video has variable frame rate, the durations may be off. We
  // should try to see if we can set the duration more accurately. Unfortunately
  // it's not given by dispInfo. One option would be to set it based on the pts
  // difference between consecutive frames, if the next frame is already
  // available.
  // Note that we used to rely on videoFormat_.frame_rate for this, but that
  // proved less accurate than FFmpeg.
  set_duration(
      av_frame, compute_safe_duration(frame_rate_avg_from_ffmpeg_, time_base_));

  // We need to assign the frame colorspace. This is crucial for proper color
  // conversion. NVCUVID stores that in the matrix_coefficients field, but
  // doesn't document the semantics of the values. Claude code generated this,
  // which seems to work. Reassuringly, the values seem to match the
  // corresponding indices in the FFmpeg enum for colorspace conversion
  // (ff_yuv2rgb_coeffs):
  // https://ffmpeg.org/doxygen/trunk/yuv2rgb_8c_source.html#l00047
  switch (video_format_.video_signal_description.matrix_coefficients) {
    case 1:
      av_frame->colorspace = AVCOL_SPC_BT709;
      break;
    case 6:
      av_frame->colorspace = AVCOL_SPC_SMPTE170M; // BT.601
      break;
    case 9:
      av_frame->colorspace = AVCOL_SPC_BT2020_NCL;
      break;
    case 10:
      av_frame->colorspace = AVCOL_SPC_BT2020_CL;
      break;
    default:
      // Default to BT.601
      av_frame->colorspace = AVCOL_SPC_SMPTE170M;
      break;
  }

  av_frame->color_range =
      video_format_.video_signal_description.video_full_range_flag
      ? AVCOL_RANGE_JPEG
      : AVCOL_RANGE_MPEG;

  // NVDEC's surface layout places the UV plane after the Y plane. For
  // NV12/P016 the Y plane has an even number of rows (NVDEC rounds up
  // internally), so we must use the rounded-up height for the UV offset.
  unsigned int even_height = round_up_to_even(height);
  av_frame->data[0] = reinterpret_cast<uint8_t*>(frame_ptr);
  av_frame->data[1] =
      reinterpret_cast<uint8_t*>(frame_ptr + (pitch * even_height));
  av_frame->data[2] = nullptr;
  av_frame->data[3] = nullptr;
  av_frame->linesize[0] = pitch;
  av_frame->linesize[1] = pitch;
  av_frame->linesize[2] = 0;
  av_frame->linesize[3] = 0;

  return av_frame;
}

void BetaCudaDeviceInterface::flush() {
  if (cpu_fallback_) {
    cpu_fallback_->flush();
    return;
  }

  // The NVCUVID docs mention that after seeking, i.e. when flush() is called,
  // we should send a packet with the CUVID_PKT_DISCONTINUITY flag. The docs
  // don't say whether this should be an empty packet, or whether it should be a
  // flag on the next non-empty packet. It doesn't matter: neither work :)
  // Sending an EOF packet, however, does work. So we do that. And we re-set the
  // eofSent_ flag to false because that's not a true EOF notification.
  send_eof_packet();
  eof_sent_ = false;

  std::queue<CUVIDPARSERDISPINFO> empty_queue;
  std::swap(ready_frames_, empty_queue);
}

UniqueAVFrame BetaCudaDeviceInterface::transfer_cpu_frame_to_gpu(
    UniqueAVFrame& cpu_frame,
    AVPixelFormat target_pix_fmt) {
  // This is called in the context of the CPU fallback: the frame was decoded on
  // the CPU, and in this function we convert that frame into NV12 or P016
  // format and send it to the GPU.
  // We do that in 2 steps:
  // - First we convert the input CPU frame into an intermediate NV12/P016 CPU
  //   frame using sws_scale.
  // - Then we allocate GPU memory and copy the CPU frame to the GPU. This
  //   is what we return.
  // Since NV12/P016 require even dimensions, the returned frame will have even
  // (rounded up) width and height, even if the original CPU frame had odd
  // dimensions.

  STD_TORCH_CHECK(cpu_frame != nullptr, "CPU frame cannot be null");
  // NV12 = 1 byte per sample, P016 = 2 bytes per sample
  STD_TORCH_CHECK(
      target_pix_fmt == AV_PIX_FMT_NV12 || target_pix_fmt == AV_PIX_FMT_P016LE,
      "targetPixFmt must be NV12 or P016LE");
  int bytes_per_sample = (target_pix_fmt == AV_PIX_FMT_P016LE) ? 2 : 1;

  int width = cpu_frame->width;
  int height = cpu_frame->height;
  int even_width = round_up_to_even(width);
  int even_height = round_up_to_even(height);

  UniqueAVFrame intermediate_cpu_frame(av_frame_alloc());
  STD_TORCH_CHECK(
      intermediate_cpu_frame != nullptr,
      "Failed to allocate intermediate CPU frame");

  intermediate_cpu_frame->format = target_pix_fmt;
  intermediate_cpu_frame->width = even_width;
  intermediate_cpu_frame->height = even_height;

  int ret = av_frame_get_buffer(intermediate_cpu_frame.get(), 0);
  STD_TORCH_CHECK(
      ret >= 0,
      "Failed to allocate intermediate CPU frame buffer: ",
      get_ffmpeg_error_string_from_error_code(ret));

  SwsConfig sws_config(
      width,
      height,
      static_cast<AVPixelFormat>(cpu_frame->format),
      cpu_frame->colorspace,
      even_width,
      even_height,
      target_pix_fmt);

  if (!sws_context_ || prev_sws_config_ != sws_config) {
    sws_context_ = create_sws_context(sws_config, SWS_BILINEAR);
    prev_sws_config_ = sws_config;
  }

  int converted_height = sws_scale(
      sws_context_.get(),
      cpu_frame->data,
      cpu_frame->linesize,
      0,
      height,
      intermediate_cpu_frame->data,
      intermediate_cpu_frame->linesize);
  STD_TORCH_CHECK(
      converted_height == even_height,
      "sws_scale failed for CPU->NV12/P016 conversion");

  int row_bytes = even_width * bytes_per_sample;
  int y_size = row_bytes * even_height;
  int uv_size = y_size / 2;
  size_t total_size = static_cast<size_t>(y_size + uv_size);

  uint8_t* cuda_buffer = nullptr;
  cudaError_t err =
      cudaMalloc(reinterpret_cast<void**>(&cuda_buffer), total_size);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "Failed to allocate CUDA memory: ",
      cudaGetErrorString(err));

  UniqueAVFrame gpu_frame(av_frame_alloc());
  STD_TORCH_CHECK(gpu_frame != nullptr, "Failed to allocate GPU AVFrame");

  gpu_frame->format = target_pix_fmt;
  gpu_frame->width = even_width;
  gpu_frame->height = even_height;
  gpu_frame->data[0] = cuda_buffer;
  gpu_frame->data[1] = cuda_buffer + y_size;
  gpu_frame->linesize[0] = row_bytes;
  gpu_frame->linesize[1] = row_bytes;

  // Note that we use cudaMemcpy2D here instead of cudaMemcpy because the
  // linesizes (strides) may be different than the widths for the input CPU
  // frame. That's precisely what cudaMemcpy2D is for.
  err = cudaMemcpy2D(
      gpu_frame->data[0],
      gpu_frame->linesize[0],
      intermediate_cpu_frame->data[0],
      intermediate_cpu_frame->linesize[0],
      row_bytes,
      even_height,
      cudaMemcpyHostToDevice);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "Failed to copy Y plane to GPU: ",
      cudaGetErrorString(err));

  err = cudaMemcpy2D(
      gpu_frame->data[1],
      gpu_frame->linesize[1],
      intermediate_cpu_frame->data[1],
      intermediate_cpu_frame->linesize[1],
      row_bytes,
      even_height / 2,
      cudaMemcpyHostToDevice);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "Failed to copy UV plane to GPU: ",
      cudaGetErrorString(err));

  ret = av_frame_copy_props(gpu_frame.get(), cpu_frame.get());
  STD_TORCH_CHECK(
      ret >= 0,
      "Failed to copy frame properties: ",
      get_ffmpeg_error_string_from_error_code(ret));

  // We need to make sure the CUDA memory is freed properly. Since we allocated
  // it ourselves, FFmpeg doesn't know how to free it. We associate a `free`
  // callback via opaque_ref that will be called by av_frame_free().
  gpu_frame->opaque_ref = av_buffer_create(
      nullptr, // data - we don't need any
      0, // data size
      cuda_buffer_free_callback, // callback triggered by av_frame_free()
      cuda_buffer, // parameter to callback
      0); // flags
  STD_TORCH_CHECK(
      gpu_frame->opaque_ref != nullptr,
      "Failed to create GPU memory cleanup reference");

  return gpu_frame;
}

void BetaCudaDeviceInterface::convert_av_frame_to_frame_output(
    UniqueAVFrame& av_frame,
    FrameOutput& frame_output,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
  if (cpu_fallback_) {
    // When the CPU fallback happens, we'll try to run the color-conversion on
    // GPU by sending those CPU frames to the GPU as NV12 or P016 (See
    // transferCpuFrameToGpu() below). However, it's not always
    // possible: NV12/P016 would downsample 4:4:4 frames and lose chroma
    // resolution, resulting in poorly decoded frames. So for those, we still
    // do the color conversion on the CPU and then send the full RGB frame to
    // the GPU.
    const AVPixFmtDescriptor* desc =
        av_pix_fmt_desc_get(static_cast<AVPixelFormat>(av_frame->format));
    bool is444 = desc && desc->log2_chroma_w == 0 && desc->log2_chroma_h == 0;
    if (is444) {
      FrameOutput cpu_frame_output;
      cpu_fallback_->convert_av_frame_to_frame_output(
          av_frame, cpu_frame_output);
      if (pre_allocated_output_tensor.has_value()) {
        torch::stable::copy_(
            pre_allocated_output_tensor.value(), cpu_frame_output.data);
        frame_output.data = pre_allocated_output_tensor.value();
      } else {
        frame_output.data = torch::stable::to(cpu_frame_output.data, device_);
      }
      if (rotation_ != Rotation::NONE) {
        apply_rotation(frame_output, pre_allocated_output_tensor);
      }
      return;
    }
  }

  // Capture original dimensions before transferCpuFrameToGpu()
  // may round them up to even.
  FrameDims original_dims(av_frame->height, av_frame->width);

  UniqueAVFrame gpu_frame;
  if (cpu_fallback_) {
    AVPixelFormat target_pix_fmt = (output_dtype_ == OutputDtype::FLOAT32)
        ? AV_PIX_FMT_P016LE
        : AV_PIX_FMT_NV12;
    gpu_frame = transfer_cpu_frame_to_gpu(av_frame, target_pix_fmt);
  } else {
    gpu_frame = std::move(av_frame);
  }

  STD_TORCH_CHECK(
      gpu_frame->format == AV_PIX_FMT_NV12 ||
          gpu_frame->format == AV_PIX_FMT_P016LE,
      "Expected NV12 or P016LE format frame");

  cudaStream_t nvdec_stream = get_current_cuda_stream(device_.index());

  auto convert_frame = [&](std::optional<torch::stable::Tensor> pre_alloc)
      -> torch::stable::Tensor {
    bool is_p016 = (gpu_frame->format == AV_PIX_FMT_P016LE);
    int bit_depth = 8;
    if (is_p016) {
      bit_depth = cpu_fallback_
          ? codec_context_->bits_per_raw_sample
          : static_cast<int>(video_format_.bit_depth_luma_minus8) + 8;
    }
    return convert_yuv_frame_to_rgb(
        gpu_frame,
        device_,
        nvdec_stream,
        pre_alloc,
        original_dims,
        is_p016,
        bit_depth,
        cached_color_matrix_);
  };

  if (rotation_ == Rotation::NONE) {
    validate_pre_allocated_tensor_shape(
        pre_allocated_output_tensor, original_dims);
    frame_output.data = convert_frame(pre_allocated_output_tensor);
  } else {
    // preAllocatedOutputTensor has post-rotation dimensions, but the
    // conversion outputs pre-rotation dimensions, so we can't use it as the
    // conversion destination or validate it against the frame shape.
    // Once we support native transforms on the NVDEC CUDA interface,
    // rotation should be handled as part of the transform pipeline instead.
    frame_output.data = convert_frame(/*preAlloc=*/std::nullopt);
    apply_rotation(frame_output, pre_allocated_output_tensor);
  }
}

void BetaCudaDeviceInterface::apply_rotation(
    FrameOutput& frame_output,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
  int k = 0;
  switch (rotation_) {
    case Rotation::CCW90:
      k = 1;
      break;
    case Rotation::ROTATE180:
      k = 2;
      break;
    case Rotation::CW90:
      k = 3;
      break;
    default:
      STD_TORCH_CHECK(false, "Unexpected rotation value");
      break;
  }
  // Apply rotation using rot90 on the H and W dims of our HWC tensor.
  // stableRot90 returns a view, so we need to make it contiguous.
  frame_output.data =
      torch::stable::contiguous(stable_rot90(frame_output.data, k, 0, 1));

  if (pre_allocated_output_tensor.has_value()) {
    torch::stable::copy_(
        pre_allocated_output_tensor.value(), frame_output.data);
    frame_output.data = pre_allocated_output_tensor.value();
  }
}

OutputDtype BetaCudaDeviceInterface::get_pre_allocation_dtype(
    OutputDtype requested_dtype) const {
  if (requested_dtype == OutputDtype::FLOAT32 &&
      surface_format_ == cudaVideoSurfaceFormat_NV12) {
    return OutputDtype::UINT8;
  }
  return requested_dtype;
}

std::string BetaCudaDeviceInterface::get_details() {
  std::string details = "NVDEC CUDA Device Interface.";
  if (cpu_fallback_) {
    details += " Using CPU fallback.";
    if (!nvcuvid_available_) {
      details += " NVCUVID not available!";
    }
  } else {
    details += " Using NVDEC.";
  }
  return details;
}

} // namespace facebook::torchcodec
