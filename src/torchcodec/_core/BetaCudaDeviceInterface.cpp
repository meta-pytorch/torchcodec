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

// The pixel formats a GPU frame can be in: the NVDEC surface formats. Frames
// decoded by NVDEC are natively in one of these; CPU-fallback frames are
// converted into one of these when they're uploaded.
// XXX is this more like "format that nvdec supports"? More than "this can only be an nvdec format"?
// XXX do we have coverage for color-conversion of all of these (444 and 444-16?)
bool is_nvdec_surface_format(int format) {
  return format == AV_PIX_FMT_NV12 || is_nvdec_16bit_surface(format) ||
      format == AV_PIX_FMT_YUV444P || format == AV_PIX_FMT_YUV444P16LE;
}

// Which of those a CPU-fallback frame should be uploaded as: whatever NVDEC
// would have produced for the same content. The depth follows the same rule as
// get_preferred_surface_format() (a 16-bit surface only buys us something for a
// float32 output on a high bit depth source), and the chroma subsampling is
// never reduced, so anything that isn't 4:2:0 goes to 4:4:4.
// XXX Should this be get_pix_fmt_for_fallback_cpu_frame() ?
AVPixelFormat fallback_upload_pix_fmt(
    const AVFrame& cpu_frame,
    OutputDtype output_dtype) {
  const AVPixFmtDescriptor* desc =
      av_pix_fmt_desc_get(static_cast<AVPixelFormat>(cpu_frame.format));
  STD_TORCH_CHECK(desc != nullptr, "Unknown pixel format on decoded frame");

  int bit_depth = desc->comp[0].depth;
  bool want_16bit = output_dtype == OutputDtype::FLOAT32 && bit_depth > 8;
  bool is_420_or_mono = desc->nb_components == 1 ||
      (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 1);

  if (is_420_or_mono) {
    return want_16bit ? nvdec_pix_fmt(/*is_p016_surface=*/true, bit_depth)
                      : AV_PIX_FMT_NV12;
  }
  return want_16bit ? AV_PIX_FMT_YUV444P16LE : AV_PIX_FMT_YUV444P;
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

void standalone_frame_free_callback(
    [[maybe_unused]] void* opaque,
    uint8_t* data) {
  delete reinterpret_cast<StandAloneFrameAttachedData*>(data);
}

class CudaContextGuard {
  // There's one CUDA context per process per device. But new threads aren't
  // bound to a context. The binding often happens automatically when calling
  // CUDA APIs (like cudaFree), but some APIs like the NVCUVID ones that we use
  // here aren't automatically binding.
  // So for a thread to be able to use NVCUVID APIs, it must have a context
  // bound to it, and we have to enforce that binding manually.
  // That's what this guard does: it calls cudaFree(nullptr), which is a common
  // near-free way to force the CUDA runtime to bind the context for the current
  // thread. And this call must happen within a device guard to make sure we're
  // binding the context of the device this interface is using.
  // We must call this guard in every public method of the interface that uses
  // NVCUVID APIs, because these methods can, in theory, be called from any
  // thread.
  // Note that none of this was an issue before when our only entry-point was
  // the SingleStreamDecoder: all the entry-points were called from the same
  // thread. Now that we have split the APIs in different blocks (PacketDecoder,
  // ColorConverter), each of these blocks can be on different threads - and
  // importantly, they can be created in the main thread (where the context is
  // bound by our call to initialize_cuda_context_with_pytorch()), but then used
  // in a different thread that doesn't have the context.
 public:
  explicit CudaContextGuard(int device_index) : device_guard_(device_index) {
    cudaFree(nullptr);
  }

 private:
  StableDeviceGuard device_guard_;
};

} // namespace

BetaCudaDeviceInterface::BetaCudaDeviceInterface(const StableDevice& device)
    : DeviceInterface(device) {
  STD_TORCH_CHECK(g_cuda_nvdec, "NvdecCudaDeviceInterface was not registered!");
  STD_TORCH_CHECK(
      device_.type() == kStableCUDA, "Unsupported device: must be CUDA");

  // Note: now that we have the CudaContextGuard, we might not need to do that
  // anymore. The comment says we need pytorch to create the context - maybe
  // that's true, but that's a very old comment now.
  initialize_cuda_context_with_pytorch(device_);

  nvcuvid_available_ = load_nvcuvid_library();
}

BetaCudaDeviceInterface::Mode BetaCudaDeviceInterface::mode() const {
  if (decoding_initialized_ && color_conversion_initialized_) {
    return Mode::Both;
  } else if (decoding_initialized_) {
    return Mode::DecoderOnly;
  } else if (color_conversion_initialized_) {
    return Mode::ColorConverterOnly;
  }
  return Mode::Uninitialized;
}

void BetaCudaDeviceInterface::initialize_color_conversion(
    const VideoStreamOptions& video_stream_options,
    [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>& transforms,
    [[maybe_unused]] const std::optional<FrameDims>& resized_output_dims) {
  // XXX We used to create a cpu device interface here - now we don't. Seems
  // like we have an invariant that if mode == color-convert-only then we don't
  // have a cpu device interface. Do we need to enforce that in some places?
  output_dtype_ = video_stream_options.output_dtype;
  color_conversion_initialized_ = true;
}

void BetaCudaDeviceInterface::initialize_video_decoding(
    const AVStream* av_stream,
    const UniqueDecodingAVFormatContext& av_format_ctx,
    const VideoStreamOptions& video_stream_options) {
  STD_TORCH_CHECK(av_stream != nullptr, "AVStream cannot be null");
  CudaContextGuard context_guard(device_.index());
  decoding_initialized_ = true;
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
    decoding_on_cpu_ = true;
    cpu_interface_ = create_device_interface(kStableCPU);
    STD_TORCH_CHECK(
        cpu_interface_ != nullptr, "Failed to create CPU device interface");
    cpu_interface_->initialize(codec_context_);
    cpu_interface_->initialize_video_decoding(
        av_stream, av_format_ctx, video_stream_options);
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
  //
  const uint8_t* seqhdr = codec_par->extradata;
  int seqhdr_data_size = codec_par->extradata_size;
  if (bitstream_filter_ && bitstream_filter_->par_out->extradata_size > 0) {
    // when a BSF is used (e.g. h264_mp4toannexb), we must pass the
    // *filtered* extradata!
    seqhdr = bitstream_filter_->par_out->extradata;
    seqhdr_data_size = bitstream_filter_->par_out->extradata_size;
  }
  if (seqhdr_data_size > 0) {
    auto seqhdr_size = std::min(
        static_cast<size_t>(seqhdr_data_size),
        sizeof(parser_ext_info_.raw_seqhdr_data));
    parser_ext_info_.format.seqhdr_data_length = seqhdr_size;
    memcpy(parser_ext_info_.raw_seqhdr_data, seqhdr, seqhdr_size);
    parser_params.pExtVideoInfo = &parser_ext_info_;
  }

  CUresult result = cuvidCreateVideoParser(&video_parser_, &parser_params);
  STD_TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create video parser: ", result);

  send_seqhdr_packet();
}

void BetaCudaDeviceInterface::send_seqhdr_packet() {
  // This must be called at parser initialization, and after each flush.
  // See https://github.com/meta-pytorch/torchcodec/pull/1503 for details.
  // FFmpeg's nvcuviddec.c does the same thing (not the nvdec.c one, because it
  // doesn't rely on the nvcuvid parser):
  // -
  // https://github.com/FFmpeg/FFmpeg/blob/d244d438c372b76d825be4527fccfd162429010a/libavcodec/cuviddec.c#L1211
  // -
  // https://github.com/FFmpeg/FFmpeg/blob/d244d438c372b76d825be4527fccfd162429010a/libavcodec/cuviddec.c#L1157
  if (parser_ext_info_.format.seqhdr_data_length == 0) {
    return;
  }
  CUVIDSOURCEDATAPACKET seq_pkt = {};
  seq_pkt.payload = parser_ext_info_.raw_seqhdr_data;
  seq_pkt.payload_size = parser_ext_info_.format.seqhdr_data_length;
  send_cuvid_packet(seq_pkt);
}

BetaCudaDeviceInterface::~BetaCudaDeviceInterface() {
  CudaContextGuard context_guard(device_.index());
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
  CudaContextGuard context_guard(device_.index());
  if (decoding_on_cpu_) {
    return cpu_interface_->send_packet(packet);
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
  CudaContextGuard context_guard(device_.index());
  if (decoding_on_cpu_) {
    return cpu_interface_->send_eof_packet();
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
  CudaContextGuard context_guard(device_.index());
  if (decoding_on_cpu_) {
    return cpu_interface_->receive_frame(av_frame);
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
  // We, OTOH, will do the color-conversion later, outside of receive_frame().
  // So we unmap here: just before mapping a new frame. At that point we know
  // that the previously-mapped frame is no longer needed:
  // - With SingleStreamDecoder, that frame was either color-converted (with a
  //   copy), or that's a frame that was discarded in SingleStreamDecoder.
  //   Either way, the underlying output surface can be safely re-used.
  // - With the "Blocks" APIs, the PacketDecoder forces a copy in
  //   make_frame_standalone().
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
  av_frame->format = nvdec_pix_fmt(
      surface_format_ == cudaVideoSurfaceFormat_P016,
      static_cast<int>(video_format_.bit_depth_luma_minus8) + 8);
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
      *av_frame,
      compute_safe_duration(frame_rate_avg_from_ffmpeg_, time_base_));

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
  // TODO_API_BREAKDOWN_CUDA P2: Check range before cast?
  av_frame->linesize[0] = static_cast<int>(pitch);
  av_frame->linesize[1] = static_cast<int>(pitch);
  av_frame->linesize[2] = 0;
  av_frame->linesize[3] = 0;

  return av_frame;
}

void BetaCudaDeviceInterface::make_frame_standalone(UniqueAVFrame& av_frame) {
  // Make the frame standalone:
  // - Crucially, we copy the frame data so that its surface can be unmapped in
  //   receive_frame() (see comment there).
  // - We put the frame in a state such that it can be safely used by a
  //   ColorConverter (i.e. a *different* instance of this
  //   BetaCudaDeviceInterface): we attach relevant metadata as the
  //   StandAloneFrameAttachedData struct, which is then used by the
  //   ColorConverter in convert_cuda_frame_to_av_frame() to perform the
  //   color-conversion correctly.
  // CPU-fallback frames are uploaded here too, so that a PacketDecoder always
  // hands out frames that live on its own device. This is the only place we do
  // that: the SingleStreamDecoder never calls this, and uploading at decode
  // time would waste bandwidth on the frames its seek loop decodes and drops.
  STD_TORCH_CHECK(
      mode() == Mode::DecoderOnly,
      "make_frame_standalone() is only valid in decoder-only mode: standalone "
      "frames are meant to be consumed by a separate ColorConverter.");
  CudaContextGuard context_guard(device_.index());

  if (decoding_on_cpu_) {
    // XXX OK so it's within transfer_cpu_frame_to_gpu that we'll set the
    // attached data - can we still do it only here? Maybe we can't because
    // transfer_cpu_frame_to_gpu() is also called from within
    // convert_av_frame_to_frame_output in the 'Both' mode?
    // How can we clean that up?
    av_frame = transfer_cpu_frame_to_gpu(*av_frame);
    return;
  }

  cudaStream_t current_stream = get_current_cuda_stream(device_.index());

  auto attached_data = new StandAloneFrameAttachedData();
  attached_data->producer_stream = current_stream;

  // The amount of bytes an NV12 image takes is:
  // num_bytes =  len(Y) + len(UV)
  //           = num_pixels + num_pixels / 2
  //           = num_pixels * 3 / 2
  //
  // where num_pixels = pitch * height, not num_pixels = width * height. The
  // pitch value also accounts for the data size (uint8 vs uint16) so this is
  // also correct for P016.
  int64_t even_height =
      static_cast<int64_t>(round_up_to_even(av_frame->height));
  int64_t pitch = static_cast<int64_t>(av_frame->linesize[0]);
  int64_t num_bytes = pitch * even_height * 3 / 2;

  auto storage =
      torch::stable::empty({num_bytes}, kStableUInt8, std::nullopt, device_);

  // TODO_API_BREAKDOWN_CUDA P1: I suspect we don't need to wait on the nvdec
  // stream here, because we can only arrive here from a path where the frame
  // has already been mapped so its data is available - worth double checking.
  cudaError_t err = cudaMemcpyAsync(
      storage.mutable_data_ptr(),
      av_frame->data[0],
      static_cast<size_t>(num_bytes),
      cudaMemcpyDeviceToDevice,
      current_stream);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "Failed to copy NVDEC surface: ",
      cudaGetErrorString(err));

  // TODO_API_BREAKDOWN_CUDA P2: Should we unmap here? Or let the next
  // receive_frame() call do it?
  // unmap_previous_frame();

  auto y_plane = static_cast<uint8_t*>(storage.mutable_data_ptr());
  av_frame->data[0] = y_plane;
  av_frame->data[1] = y_plane + (pitch * even_height);

  // TODO_API_BREAKDOWN_CUDA P2: We don't *really* need to std::move it I
  // guess?
  attached_data->storage = std::move(storage);

  av_frame->opaque_ref = av_buffer_create(
      reinterpret_cast<uint8_t*>(attached_data),
      sizeof(StandAloneFrameAttachedData),
      standalone_frame_free_callback,
      nullptr,
      0);
}

// XXX We may not need this anymore?
bool BetaCudaDeviceInterface::is_device_frame(
    [[maybe_unused]] const UniqueAVFrame& av_frame) const {
  // make_frame_standalone() uploads CPU-fallback frames, and it is the only
  // path through which frames leave this interface as standalone frames.
  return true;
}

void BetaCudaDeviceInterface::flush() {
  CudaContextGuard context_guard(device_.index());
  if (decoding_on_cpu_) {
    cpu_interface_->flush();
    return;
  }

  // The NVCUVID docs mention that after seeking, i.e. when flush() is called,
  // we should send a packet with the CUVID_PKT_DISCONTINUITY flag. The docs
  // don't say whether this should be an empty packet, or whether it should be
  // a flag on the next non-empty packet. It doesn't matter: neither work :)
  // Sending an EOF packet, however, does work. So we do that. And we re-set
  // the eofSent_ flag to false because that's not a true EOF notification.
  send_eof_packet();
  eof_sent_ = false;

  std::queue<CUVIDPARSERDISPINFO> empty_queue;
  std::swap(ready_frames_, empty_queue);

  send_seqhdr_packet();
}

// XXX Nit: rename this into upload_cpu_frame_to_gpu?
UniqueAVFrame BetaCudaDeviceInterface::transfer_cpu_frame_to_gpu(
    const AVFrame& cpu_frame) {
  // This is called in the context of the CPU fallback: the frame was decoded
  // on the CPU, and in this function we convert that frame into an NVDEC
  // surface format and send it to the GPU.
  // We do that in 2 steps:
  // - First we convert the input CPU frame into an intermediate CPU frame in
  //   the target format using sws_scale.
  // - Then we allocate GPU memory and copy the CPU frame to the GPU. This
  //   is what we return.
  // XXX Is it really semi-planar formats that require even dimensions? Is it
  // truly what characterizes them? Or is it just that they are subsampled (i.e. not 444)?
  // Since the semi-planar formats require even dimensions, the returned frame
  // will have even (rounded up) width and height for those, even if the
  // original CPU frame had odd dimensions.
  AVPixelFormat target_pix_fmt =
      fallback_upload_pix_fmt(cpu_frame, output_dtype_);
  const AVPixFmtDescriptor* target_desc = av_pix_fmt_desc_get(target_pix_fmt);
  STD_TORCH_CHECK(target_desc != nullptr, "Unknown target pixel format");

  bool is_444 = target_desc->log2_chroma_w == 0;
  int num_planes = is_444 ? 3 : 2;
  int bytes_per_sample = target_desc->comp[0].depth > 8 ? 2 : 1;

  int width = cpu_frame.width;
  int height = cpu_frame.height;
  int target_width = is_444 ? width : round_up_to_even(width);
  int target_height = is_444 ? height : round_up_to_even(height);

  UniqueAVFrame intermediate_cpu_frame(av_frame_alloc());
  STD_TORCH_CHECK(
      intermediate_cpu_frame != nullptr,
      "Failed to allocate intermediate CPU frame");

  intermediate_cpu_frame->format = target_pix_fmt;
  intermediate_cpu_frame->width = target_width;
  intermediate_cpu_frame->height = target_height;

  int ret = av_frame_get_buffer(intermediate_cpu_frame.get(), 0);
  STD_TORCH_CHECK(
      ret >= 0,
      "Failed to allocate intermediate CPU frame buffer: ",
      get_ffmpeg_error_string_from_error_code(ret));

  SwsConfig sws_config(
      width,
      height,
      static_cast<AVPixelFormat>(cpu_frame.format),
      cpu_frame.colorspace,
      target_width,
      target_height,
      target_pix_fmt);

  if (!sws_context_ || prev_sws_config_ != sws_config) {
    sws_context_ = create_sws_context(sws_config, SWS_BILINEAR);
    prev_sws_config_ = sws_config;
  }

  int converted_height = sws_scale(
      sws_context_.get(),
      cpu_frame.data,
      cpu_frame.linesize,
      0,
      height,
      intermediate_cpu_frame->data,
      intermediate_cpu_frame->linesize);
  STD_TORCH_CHECK(
      converted_height == target_height,
      "sws_scale failed for the CPU-fallback upload conversion");

  // The chroma plane of a semi-planar 4:2:0 frame carries interleaved UV pairs,
  // so it's as wide as the luma plane but half as tall.
  int row_bytes = target_width * bytes_per_sample;
  int plane_heights[3] = {
      target_height, is_444 ? target_height : target_height / 2, target_height};

  int64_t plane_offsets[3] = {0, 0, 0};
  int64_t total_bytes = 0;
  for (int p = 0; p < num_planes; ++p) {
    plane_offsets[p] = total_bytes;
    total_bytes += static_cast<int64_t>(row_bytes) * plane_heights[p];
  }

  CudaContextGuard context_guard(device_.index());
  cudaStream_t current_stream = get_current_cuda_stream(device_.index());
  auto storage =
      torch::stable::empty({total_bytes}, kStableUInt8, std::nullopt, device_);
  auto storage_ptr = static_cast<uint8_t*>(storage.mutable_data_ptr());

  UniqueAVFrame gpu_frame(av_frame_alloc());
  STD_TORCH_CHECK(gpu_frame != nullptr, "Failed to allocate GPU AVFrame");

  gpu_frame->format = target_pix_fmt;
  gpu_frame->width = target_width;
  gpu_frame->height = target_height;

  // XXX Why do we need seperate uploads per-plane?
  for (int p = 0; p < num_planes; ++p) {
    gpu_frame->data[p] = storage_ptr + plane_offsets[p];
    gpu_frame->linesize[p] = row_bytes;

    // Note that we use cudaMemcpy2D here instead of cudaMemcpy because the
    // linesizes (strides) may be different than the widths for the input CPU
    // frame. That's precisely what cudaMemcpy2D is for.
    cudaError_t err = cudaMemcpy2D(
        gpu_frame->data[p],
        gpu_frame->linesize[p],
        intermediate_cpu_frame->data[p],
        intermediate_cpu_frame->linesize[p],
        row_bytes,
        plane_heights[p],
        cudaMemcpyHostToDevice);
    STD_TORCH_CHECK(
        err == cudaSuccess,
        "Failed to copy plane ",
        p,
        " to GPU: ",
        cudaGetErrorString(err));
  }

  ret = av_frame_copy_props(gpu_frame.get(), &cpu_frame);
  STD_TORCH_CHECK(
      ret >= 0,
      "Failed to copy frame properties: ",
      get_ffmpeg_error_string_from_error_code(ret));

  // XXX OK but this path can be reached from the Both() mode, i.e.
  // SingleStreamDecoder. And those frames Don't need to be 'StandAlone' -
  // although they do need the attached data both for stream sync and for storage.
  // What would be a better name? Maybe we should split some logic to make it
  // more atomic.
  // av_frame_copy_props() copies opaque_ref, so this must come after it. The
  // attached storage is what keeps the GPU memory alive.
  auto attached_data = new StandAloneFrameAttachedData();
  attached_data->producer_stream = current_stream;
  attached_data->storage = std::move(storage);
  gpu_frame->opaque_ref = av_buffer_create(
      reinterpret_cast<uint8_t*>(attached_data),
      sizeof(StandAloneFrameAttachedData),
      standalone_frame_free_callback,
      nullptr,
      0);
  STD_TORCH_CHECK(
      gpu_frame->opaque_ref != nullptr,
      "Failed to create GPU memory cleanup reference");

  return gpu_frame;
}

void BetaCudaDeviceInterface::convert_av_frame_to_frame_output(
    const AVFrame& av_frame,
    FrameOutput& frame_output,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
  CudaContextGuard context_guard(device_.index());

  // In ColorConverterOnly mode the frame comes from a PacketDecoder, which
  // already uploaded it. Here, we're the interface that decoded it, so we know
  // first-hand whether it needs uploading.
  // XXX Should have a clear comment about why this is deliberately done here
  // and not in receive_frame().
  bool needs_upload = mode() == Mode::Both && decoding_on_cpu_;

  // Capture original dimensions before transfer_cpu_frame_to_gpu()
  // may round them up to even.
  FrameDims original_dims(av_frame.height, av_frame.width);

  UniqueAVFrame transferred_frame;
  if (needs_upload) {
    transferred_frame = transfer_cpu_frame_to_gpu(av_frame);
  }
  const AVFrame& gpu_frame = needs_upload ? *transferred_frame : av_frame;

  STD_TORCH_CHECK(
      is_nvdec_surface_format(gpu_frame.format),
      "Expected the frame to be in an NVDEC surface format, got ",
      av_get_pix_fmt_name(static_cast<AVPixelFormat>(gpu_frame.format)));

  cudaStream_t producer_stream;
  // XXX it'll work because in Both() mode the upload will set the upload stream
  // as the current_stream, so the else claude below will implicitly match it,
  // but it seems brittle. Maybe we should always set producer_stream to
  // attached_data->producer_stream if it exists.
  if (mode() == Mode::ColorConverterOnly) {
    STD_TORCH_CHECK(
        av_frame.opaque_ref != nullptr,
        "ColorConverter received a non-standalone frame; frames fed to a "
        "standalone ColorConverter must come from a PacketDecoder.");
    auto attached_data = reinterpret_cast<StandAloneFrameAttachedData*>(
        av_frame.opaque_ref->data);
    producer_stream = attached_data->producer_stream;
  } else {
    producer_stream = get_current_cuda_stream(device_.index());
  }

  auto convert_frame = [&](std::optional<torch::stable::Tensor> pre_alloc)
      -> torch::stable::Tensor {
    return convert_yuv_frame_to_rgb(
        gpu_frame,
        device_,
        producer_stream,
        pre_alloc,
        original_dims,
        static_cast<AVPixelFormat>(gpu_frame.format),
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

// XXX Might want a TORCH_CHECK(mode == BOTH)???
OutputDtype BetaCudaDeviceInterface::get_pre_allocation_dtype(
    OutputDtype requested_dtype) const {
  if (requested_dtype != OutputDtype::FLOAT32) {
    return requested_dtype;
  }

  // Color conversion produces uint16 iff the surface it reads is 16-bit, which
  // for a fallback stream is decided by the source's own depth (see
  // fallback_upload_pix_fmt()).
  bool is_16bit_surface;
  if (decoding_on_cpu_) {
    const AVPixFmtDescriptor* desc =
        av_pix_fmt_desc_get(codec_context_->pix_fmt);
    is_16bit_surface = desc != nullptr && desc->comp[0].depth > 8;
  } else {
    is_16bit_surface = surface_format_ != cudaVideoSurfaceFormat_NV12;
  }
  return is_16bit_surface ? OutputDtype::FLOAT32 : OutputDtype::UINT8;
}

std::string BetaCudaDeviceInterface::get_details() {
  std::string details = "NVDEC CUDA Device Interface.";
  if (decoding_on_cpu_) {
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
