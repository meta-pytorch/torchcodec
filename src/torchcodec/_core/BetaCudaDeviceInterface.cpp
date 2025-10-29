// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>
#include <mutex>
#include <vector>

#include "src/torchcodec/_core/BetaCudaDeviceInterface.h"

#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/NVDECCache.h"

#include "src/torchcodec/_core/NVCUVIDRuntimeLoader.h"
#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

namespace {

static bool g_cuda_beta = registerDeviceInterface(
    DeviceInterfaceKey(torch::kCUDA, /*variant=*/"beta"),
    [](const torch::Device& device) {
      return new BetaCudaDeviceInterface(device);
    });

static int CUDAAPI
pfnSequenceCallback(void* pUserData, CUVIDEOFORMAT* videoFormat) {
  auto decoder = static_cast<BetaCudaDeviceInterface*>(pUserData);
  return decoder->streamPropertyChange(videoFormat);
}

static int CUDAAPI
pfnDecodePictureCallback(void* pUserData, CUVIDPICPARAMS* picParams) {
  auto decoder = static_cast<BetaCudaDeviceInterface*>(pUserData);
  return decoder->frameReadyForDecoding(picParams);
}

static int CUDAAPI
pfnDisplayPictureCallback(void* pUserData, CUVIDPARSERDISPINFO* dispInfo) {
  auto decoder = static_cast<BetaCudaDeviceInterface*>(pUserData);
  return decoder->frameReadyInDisplayOrder(dispInfo);
}

static UniqueCUvideodecoder createDecoder(CUVIDEOFORMAT* videoFormat) {
  // Decoder creation parameters, most are taken from DALI
  CUVIDDECODECREATEINFO decoderParams = {};
  decoderParams.bitDepthMinus8 = videoFormat->bit_depth_luma_minus8;
  decoderParams.ChromaFormat = videoFormat->chroma_format;
  // We explicitly request NV12 format, which means 10bit videos will be
  // automatically converted to 8bits by NVDEC itself. That is, the raw frames
  // we get back from cuvidMapVideoFrame will already be in 8bit format.  We
  // won't need to do the conversion ourselves, so that's a lot easier.
  // In the ffmpeg CUDA interface, we have to do the 10 -> 8bits conversion
  // ourselves later in convertAVFrameToFrameOutput(), because FFmpeg explicitly
  // requests 10 or 16bits output formats for >8-bit videos!
  // https://github.com/FFmpeg/FFmpeg/blob/e05f8acabff468c1382277c1f31fa8e9d90c3202/libavcodec/nvdec.c#L376-L403
  decoderParams.OutputFormat = cudaVideoSurfaceFormat_NV12;
  decoderParams.ulCreationFlags = cudaVideoCreate_Default;
  decoderParams.CodecType = videoFormat->codec;
  decoderParams.ulHeight = videoFormat->coded_height;
  decoderParams.ulWidth = videoFormat->coded_width;
  decoderParams.ulMaxHeight = videoFormat->coded_height;
  decoderParams.ulMaxWidth = videoFormat->coded_width;
  decoderParams.ulTargetHeight =
      videoFormat->display_area.bottom - videoFormat->display_area.top;
  decoderParams.ulTargetWidth =
      videoFormat->display_area.right - videoFormat->display_area.left;
  decoderParams.ulNumDecodeSurfaces = videoFormat->min_num_decode_surfaces;
  // We should only ever need 1 output surface, since we process frames
  // sequentially, and we always unmap the previous frame before mapping a new
  // one.
  // TODONVDEC P3: set this to 2, allow for 2 frames to be mapped at a time, and
  // benchmark to see if this makes any difference.
  decoderParams.ulNumOutputSurfaces = 1;
  decoderParams.display_area.left = videoFormat->display_area.left;
  decoderParams.display_area.right = videoFormat->display_area.right;
  decoderParams.display_area.top = videoFormat->display_area.top;
  decoderParams.display_area.bottom = videoFormat->display_area.bottom;

  CUvideodecoder* decoder = new CUvideodecoder();
  CUresult result = cuvidCreateDecoder(decoder, &decoderParams);
  TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create NVDEC decoder: ", result);
  return UniqueCUvideodecoder(decoder, CUvideoDecoderDeleter{});
}

std::optional<cudaVideoChromaFormat> validateChromaSupport(
    const AVPixFmtDescriptor* desc) {
  // Return the corresponding cudaVideoChromaFormat if supported, std::nullopt
  // otherwise.
  TORCH_CHECK(desc != nullptr, "desc can't be null");

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

std::optional<cudaVideoCodec> validateCodecSupport(AVCodecID codecId) {
  // Return the corresponding cudaVideoCodec if supported, std::nullopt
  // otherwise
  // Note that we currently return nullopt (and thus fallback to CPU) for some
  // codecs that are technically supported by NVDEC, see comment below.
  switch (codecId) {
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

bool nativeNVDECSupport(const SharedAVCodecContext& codecContext) {
  // Return true iff the input video stream is supported by our NVDEC
  // implementation.

  auto codecType = validateCodecSupport(codecContext->codec_id);
  if (!codecType.has_value()) {
    return false;
  }

  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(codecContext->pix_fmt);
  if (!desc) {
    return false;
  }

  auto chromaFormat = validateChromaSupport(desc);
  if (!chromaFormat.has_value()) {
    return false;
  }

  auto caps = CUVIDDECODECAPS{};
  caps.eCodecType = codecType.value();
  caps.eChromaFormat = chromaFormat.value();
  caps.nBitDepthMinus8 = desc->comp[0].depth - 8;

  CUresult result = cuvidGetDecoderCaps(&caps);
  if (result != CUDA_SUCCESS) {
    return false;
  }

  if (!caps.bIsSupported) {
    return false;
  }

  auto coded_width = static_cast<unsigned int>(codecContext->coded_width);
  auto coded_height = static_cast<unsigned int>(codecContext->coded_height);
  if (coded_width < static_cast<unsigned int>(caps.nMinWidth) ||
      coded_height < static_cast<unsigned int>(caps.nMinHeight) ||
      coded_width > caps.nMaxWidth || coded_height > caps.nMaxHeight) {
    return false;
  }

  // See nMaxMBCount in cuviddec.h
  constexpr unsigned int macroblockConstant = 256;
  if (coded_width * coded_height / macroblockConstant > caps.nMaxMBCount) {
    return false;
  }

  // We'll set the decoderParams.OutputFormat to NV12, so we need to make
  // sure it's actually supported.
  // TODO: If this fail, we could consider decoding to something else than NV12
  // (like cudaVideoSurfaceFormat_P016) instead of falling back to CPU. This is
  // what FFmpeg does.
  bool supportsNV12Output =
      (caps.nOutputFormatMask >> cudaVideoSurfaceFormat_NV12) & 1;
  if (!supportsNV12Output) {
    return false;
  }

  return true;
}

// Callback for freeing CUDA memory associated with AVFrame see where it's used
// for more details.
void cudaBufferFreeCallback(void* opaque, [[maybe_unused]] uint8_t* data) {
  cudaFree(opaque);
}

} // namespace

BetaCudaDeviceInterface::BetaCudaDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_cuda_beta, "BetaCudaDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());

  initializeCudaContextWithPytorch(device_);
  nppCtx_ = getNppStreamContext(device_);

  nvcuvidAvailable_ = loadNVCUVIDLibrary();
}

BetaCudaDeviceInterface::~BetaCudaDeviceInterface() {
  if (decoder_) {
    // DALI doesn't seem to do any particular cleanup of the decoder before
    // sending it to the cache, so we probably don't need to do anything either.
    // Just to be safe, we flush.
    // What happens to those decode surfaces that haven't yet been mapped is
    // unclear.
    flush();
    unmapPreviousFrame();
    NVDECCache::getCache(device_).returnDecoder(
        &videoFormat_, std::move(decoder_));
  }

  if (videoParser_) {
    cuvidDestroyVideoParser(videoParser_);
    videoParser_ = nullptr;
  }

  returnNppStreamContextToCache(device_, std::move(nppCtx_));
}

void BetaCudaDeviceInterface::initialize(
    const AVStream* avStream,
    const UniqueDecodingAVFormatContext& avFormatCtx,
    [[maybe_unused]] const SharedAVCodecContext& codecContext) {
  if (!nvcuvidAvailable_ || !nativeNVDECSupport(codecContext)) {
    cpuFallback_ = createDeviceInterface(torch::kCPU);
    TORCH_CHECK(
        cpuFallback_ != nullptr, "Failed to create CPU device interface");
    cpuFallback_->initialize(avStream, avFormatCtx, codecContext);
    cpuFallback_->initializeVideo(
        VideoStreamOptions(),
        {},
        /*resizedOutputDims=*/std::nullopt);
    // We'll always use the CPU fallback from now on, so we can return early.
    return;
  }

  TORCH_CHECK(avStream != nullptr, "AVStream cannot be null");
  timeBase_ = avStream->time_base;
  frameRateAvgFromFFmpeg_ = avStream->r_frame_rate;

  const AVCodecParameters* codecPar = avStream->codecpar;
  TORCH_CHECK(codecPar != nullptr, "CodecParameters cannot be null");

  initializeBSF(codecPar, avFormatCtx);

  // Create parser. Default values that aren't obvious are taken from DALI.
  CUVIDPARSERPARAMS parserParams = {};
  auto codecType = validateCodecSupport(codecPar->codec_id);
  TORCH_CHECK(
      codecType.has_value(),
      "This should never happen, we should be using the CPU fallback by now. Please report a bug.");
  parserParams.CodecType = codecType.value();
  parserParams.ulMaxNumDecodeSurfaces = 8;
  parserParams.ulMaxDisplayDelay = 0;
  // Callback setup, all are triggered by the parser within a call
  // to cuvidParseVideoData
  parserParams.pUserData = this;
  parserParams.pfnSequenceCallback = pfnSequenceCallback;
  parserParams.pfnDecodePicture = pfnDecodePictureCallback;
  parserParams.pfnDisplayPicture = pfnDisplayPictureCallback;

  CUresult result = cuvidCreateVideoParser(&videoParser_, &parserParams);
  TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create video parser: ", result);
}

void BetaCudaDeviceInterface::initializeBSF(
    const AVCodecParameters* codecPar,
    const UniqueDecodingAVFormatContext& avFormatCtx) {
  // Setup bit stream filters (BSF):
  // https://ffmpeg.org/doxygen/7.0/group__lavc__bsf.html
  // This is only needed for some formats, like H264 or HEVC.

  TORCH_CHECK(codecPar != nullptr, "codecPar cannot be null");
  TORCH_CHECK(avFormatCtx != nullptr, "AVFormatContext cannot be null");
  TORCH_CHECK(
      avFormatCtx->iformat != nullptr,
      "AVFormatContext->iformat cannot be null");
  std::string filterName;

  // Matching logic is taken from DALI
  switch (codecPar->codec_id) {
    case AV_CODEC_ID_H264: {
      const std::string formatName = avFormatCtx->iformat->long_name
          ? avFormatCtx->iformat->long_name
          : "";

      if (formatName == "QuickTime / MOV" ||
          formatName == "FLV (Flash Video)" ||
          formatName == "Matroska / WebM" || formatName == "raw H.264 video") {
        filterName = "h264_mp4toannexb";
      }
      break;
    }

    case AV_CODEC_ID_HEVC: {
      const std::string formatName = avFormatCtx->iformat->long_name
          ? avFormatCtx->iformat->long_name
          : "";

      if (formatName == "QuickTime / MOV" ||
          formatName == "FLV (Flash Video)" ||
          formatName == "Matroska / WebM" || formatName == "raw HEVC video") {
        filterName = "hevc_mp4toannexb";
      }
      break;
    }
    case AV_CODEC_ID_MPEG4: {
      const std::string formatName =
          avFormatCtx->iformat->name ? avFormatCtx->iformat->name : "";
      if (formatName == "avi") {
        filterName = "mpeg4_unpack_bframes";
      }
      break;
    }

    default:
      // No bitstream filter needed for other codecs
      break;
  }

  if (filterName.empty()) {
    // Only initialize BSF if we actually need one
    return;
  }

  const AVBitStreamFilter* avBSF = av_bsf_get_by_name(filterName.c_str());
  TORCH_CHECK(
      avBSF != nullptr, "Failed to find bitstream filter: ", filterName);

  AVBSFContext* avBSFContext = nullptr;
  int retVal = av_bsf_alloc(avBSF, &avBSFContext);
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to allocate bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  bitstreamFilter_.reset(avBSFContext);

  retVal = avcodec_parameters_copy(bitstreamFilter_->par_in, codecPar);
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to copy codec parameters: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  retVal = av_bsf_init(bitstreamFilter_.get());
  TORCH_CHECK(
      retVal == AVSUCCESS,
      "Failed to initialize bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));
}

// This callback is called by the parser within cuvidParseVideoData when there
// is a change in the stream's properties (like resolution change), as specified
// by CUVIDEOFORMAT. Particularly (but not just!), this is called at the very
// start of the stream.
// TODONVDEC P1: Code below mostly assume this is called only once at the start,
// we should handle the case of multiple calls. Probably need to flush buffers,
// etc.
int BetaCudaDeviceInterface::streamPropertyChange(CUVIDEOFORMAT* videoFormat) {
  TORCH_CHECK(videoFormat != nullptr, "Invalid video format");

  videoFormat_ = *videoFormat;

  if (videoFormat_.min_num_decode_surfaces == 0) {
    // Same as DALI's fallback
    videoFormat_.min_num_decode_surfaces = 20;
  }

  if (!decoder_) {
    decoder_ = NVDECCache::getCache(device_).getDecoder(videoFormat);

    if (!decoder_) {
      // TODONVDEC P2: consider re-configuring an existing decoder instead of
      // re-creating one. See docs, see DALI. Re-configuration doesn't seem to
      // be enabled in DALI by default.
      decoder_ = createDecoder(videoFormat);
    }

    TORCH_CHECK(decoder_, "Failed to get or create decoder");
  }

  // DALI also returns min_num_decode_surfaces from this function. This
  // instructs the parser to reset its ulMaxNumDecodeSurfaces field to this
  // value.
  return static_cast<int>(videoFormat_.min_num_decode_surfaces);
}

// Moral equivalent of avcodec_send_packet(). Here, we pass the AVPacket down to
// the NVCUVID parser.
int BetaCudaDeviceInterface::sendPacket(ReferenceAVPacket& packet) {
  if (cpuFallback_) {
    return cpuFallback_->sendPacket(packet);
  }

  TORCH_CHECK(
      packet.get() && packet->data && packet->size > 0,
      "sendPacket received an empty packet, this is unexpected, please report.");

  // Apply BSF if needed. We want applyBSF to return a *new* filtered packet, or
  // the original one if no BSF is needed. This new filtered packet must be
  // allocated outside of applyBSF: if it were allocated inside applyBSF, it
  // would be destroyed at the end of the function, leaving us with a dangling
  // reference.
  AutoAVPacket filteredAutoPacket;
  ReferenceAVPacket filteredPacket(filteredAutoPacket);
  ReferenceAVPacket& packetToSend = applyBSF(packet, filteredPacket);

  CUVIDSOURCEDATAPACKET cuvidPacket = {};
  cuvidPacket.payload = packetToSend->data;
  cuvidPacket.payload_size = packetToSend->size;
  cuvidPacket.flags = CUVID_PKT_TIMESTAMP;
  cuvidPacket.timestamp = packetToSend->pts;

  return sendCuvidPacket(cuvidPacket);
}

int BetaCudaDeviceInterface::sendEOFPacket() {
  if (cpuFallback_) {
    return cpuFallback_->sendEOFPacket();
  }

  CUVIDSOURCEDATAPACKET cuvidPacket = {};
  cuvidPacket.flags = CUVID_PKT_ENDOFSTREAM;
  eofSent_ = true;

  return sendCuvidPacket(cuvidPacket);
}

int BetaCudaDeviceInterface::sendCuvidPacket(
    CUVIDSOURCEDATAPACKET& cuvidPacket) {
  CUresult result = cuvidParseVideoData(videoParser_, &cuvidPacket);
  return result == CUDA_SUCCESS ? AVSUCCESS : AVERROR_EXTERNAL;
}

ReferenceAVPacket& BetaCudaDeviceInterface::applyBSF(
    ReferenceAVPacket& packet,
    ReferenceAVPacket& filteredPacket) {
  if (!bitstreamFilter_) {
    return packet;
  }

  int retVal = av_bsf_send_packet(bitstreamFilter_.get(), packet.get());
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to send packet to bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  // TODO P1: the docs mention there can theoretically be multiple output
  // packets for a single input, i.e. we may need to call av_bsf_receive_packet
  // more than once. We should figure out whether that applies to the BSF we're
  // using.
  retVal = av_bsf_receive_packet(bitstreamFilter_.get(), filteredPacket.get());
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to receive packet from bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  return filteredPacket;
}

// Parser triggers this callback within cuvidParseVideoData when a frame is
// ready to be decoded, i.e. the parser received all the necessary packets for a
// given frame. It means we can send that frame to be decoded by the hardware
// NVDEC decoder by calling cuvidDecodePicture which is non-blocking.
int BetaCudaDeviceInterface::frameReadyForDecoding(CUVIDPICPARAMS* picParams) {
  TORCH_CHECK(picParams != nullptr, "Invalid picture parameters");
  TORCH_CHECK(decoder_, "Decoder not initialized before picture decode");
  // Send frame to be decoded by NVDEC - non-blocking call.
  CUresult result = cuvidDecodePicture(*decoder_.get(), picParams);

  // Yes, you're reading that right, 0 means error, 1 means success
  return (result == CUDA_SUCCESS);
}

int BetaCudaDeviceInterface::frameReadyInDisplayOrder(
    CUVIDPARSERDISPINFO* dispInfo) {
  readyFrames_.push(*dispInfo);
  return 1; // success
}

// Moral equivalent of avcodec_receive_frame().
int BetaCudaDeviceInterface::receiveFrame(UniqueAVFrame& avFrame) {
  if (cpuFallback_) {
    return cpuFallback_->receiveFrame(avFrame);
  }

  if (readyFrames_.empty()) {
    // No frame found, instruct caller to try again later after sending more
    // packets, or to stop if EOF was already sent.
    return eofSent_ ? AVERROR_EOF : AVERROR(EAGAIN);
  }

  CUVIDPARSERDISPINFO dispInfo = readyFrames_.front();
  readyFrames_.pop();

  CUVIDPROCPARAMS procParams = {};
  procParams.progressive_frame = dispInfo.progressive_frame;
  procParams.top_field_first = dispInfo.top_field_first;
  procParams.unpaired_field = dispInfo.repeat_first_field < 0;
  // We set the NVDEC stream to the current stream. It will be waited upon by
  // the NPP stream before any color conversion.
  // Re types: we get a cudaStream_t from PyTorch but it's interchangeable with
  // CUstream
  procParams.output_stream = reinterpret_cast<CUstream>(
      at::cuda::getCurrentCUDAStream(device_.index()).stream());

  CUdeviceptr framePtr = 0;
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
  unmapPreviousFrame();
  CUresult result = cuvidMapVideoFrame(
      *decoder_.get(), dispInfo.picture_index, &framePtr, &pitch, &procParams);
  if (result != CUDA_SUCCESS) {
    return AVERROR_EXTERNAL;
  }
  previouslyMappedFrame_ = framePtr;

  avFrame = convertCudaFrameToAVFrame(framePtr, pitch, dispInfo);

  return AVSUCCESS;
}

void BetaCudaDeviceInterface::unmapPreviousFrame() {
  if (previouslyMappedFrame_ == 0) {
    return;
  }
  CUresult result =
      cuvidUnmapVideoFrame(*decoder_.get(), previouslyMappedFrame_);
  TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to unmap previous frame: ", result);
  previouslyMappedFrame_ = 0;
}

UniqueAVFrame BetaCudaDeviceInterface::convertCudaFrameToAVFrame(
    CUdeviceptr framePtr,
    unsigned int pitch,
    const CUVIDPARSERDISPINFO& dispInfo) {
  TORCH_CHECK(framePtr != 0, "Invalid CUDA frame pointer");

  // Get frame dimensions from video format display area (not coded dimensions)
  // This matches DALI's approach and avoids padding issues
  int width = videoFormat_.display_area.right - videoFormat_.display_area.left;
  int height = videoFormat_.display_area.bottom - videoFormat_.display_area.top;

  TORCH_CHECK(width > 0 && height > 0, "Invalid frame dimensions");
  TORCH_CHECK(
      pitch >= static_cast<unsigned int>(width), "Pitch must be >= width");

  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame.get() != nullptr, "Failed to allocate AVFrame");

  avFrame->width = width;
  avFrame->height = height;
  avFrame->format = AV_PIX_FMT_CUDA;
  avFrame->pts = dispInfo.timestamp;

  // TODONVDEC P2: We compute the duration based on average frame rate info, so
  // so if the video has variable frame rate, the durations may be off. We
  // should try to see if we can set the duration more accurately. Unfortunately
  // it's not given by dispInfo. One option would be to set it based on the pts
  // difference between consecutive frames, if the next frame is already
  // available.
  // Note that we used to rely on videoFormat_.frame_rate for this, but that
  // proved less accurate than FFmpeg.
  setDuration(avFrame, computeSafeDuration(frameRateAvgFromFFmpeg_, timeBase_));

  // We need to assign the frame colorspace. This is crucial for proper color
  // conversion. NVCUVID stores that in the matrix_coefficients field, but
  // doesn't document the semantics of the values. Claude code generated this,
  // which seems to work. Reassuringly, the values seem to match the
  // corresponding indices in the FFmpeg enum for colorspace conversion
  // (ff_yuv2rgb_coeffs):
  // https://ffmpeg.org/doxygen/trunk/yuv2rgb_8c_source.html#l00047
  switch (videoFormat_.video_signal_description.matrix_coefficients) {
    case 1:
      avFrame->colorspace = AVCOL_SPC_BT709;
      break;
    case 6:
      avFrame->colorspace = AVCOL_SPC_SMPTE170M; // BT.601
      break;
    default:
      // Default to BT.601
      avFrame->colorspace = AVCOL_SPC_SMPTE170M;
      break;
  }

  avFrame->color_range =
      videoFormat_.video_signal_description.video_full_range_flag
      ? AVCOL_RANGE_JPEG
      : AVCOL_RANGE_MPEG;

  // Below: Ask Claude. I'm not going to even pretend.
  avFrame->data[0] = reinterpret_cast<uint8_t*>(framePtr);
  avFrame->data[1] = reinterpret_cast<uint8_t*>(framePtr + (pitch * height));
  avFrame->data[2] = nullptr;
  avFrame->data[3] = nullptr;
  avFrame->linesize[0] = pitch;
  avFrame->linesize[1] = pitch;
  avFrame->linesize[2] = 0;
  avFrame->linesize[3] = 0;

  return avFrame;
}

void BetaCudaDeviceInterface::flush() {
  if (cpuFallback_) {
    cpuFallback_->flush();
    return;
  }

  // The NVCUVID docs mention that after seeking, i.e. when flush() is called,
  // we should send a packet with the CUVID_PKT_DISCONTINUITY flag. The docs
  // don't say whether this should be an empty packet, or whether it should be a
  // flag on the next non-empty packet. It doesn't matter: neither work :)
  // Sending an EOF packet, however, does work. So we do that. And we re-set the
  // eofSent_ flag to false because that's not a true EOF notification.
  sendEOFPacket();
  eofSent_ = false;

  std::queue<CUVIDPARSERDISPINFO> emptyQueue;
  std::swap(readyFrames_, emptyQueue);
}

UniqueAVFrame BetaCudaDeviceInterface::transferCpuFrameToGpuNV12(
    UniqueAVFrame& cpuFrame) {
  // This is called in the context of the CPU fallback: the frame was decoded on
  // the CPU, and in this function we convert that frame into NV12 format and
  // send it to the GPU.
  // We do that in 2 steps:
  // - First we convert the input CPU frame into an intermediate NV12 CPU frame
  //   using sws_scale.
  // - Then we allocate GPU memory and copy the NV12 CPU frame to the GPU. This
  //   is what we return

  TORCH_CHECK(cpuFrame != nullptr, "CPU frame cannot be null");

  int width = cpuFrame->width;
  int height = cpuFrame->height;

  // intermediate NV12 CPU frame. It's not on the GPU yet.
  UniqueAVFrame nv12CpuFrame(av_frame_alloc());
  TORCH_CHECK(nv12CpuFrame != nullptr, "Failed to allocate NV12 CPU frame");

  nv12CpuFrame->format = AV_PIX_FMT_NV12;
  nv12CpuFrame->width = width;
  nv12CpuFrame->height = height;

  int ret = av_frame_get_buffer(nv12CpuFrame.get(), 0);
  TORCH_CHECK(
      ret >= 0,
      "Failed to allocate NV12 CPU frame buffer: ",
      getFFMPEGErrorStringFromErrorCode(ret));

  SwsFrameContext swsFrameContext(
      width,
      height,
      static_cast<AVPixelFormat>(cpuFrame->format),
      width,
      height);

  if (!swsContext_ || prevSwsFrameContext_ != swsFrameContext) {
    swsContext_ = createSwsContext(
        swsFrameContext, cpuFrame->colorspace, AV_PIX_FMT_NV12, SWS_BILINEAR);
    prevSwsFrameContext_ = swsFrameContext;
  }

  int convertedHeight = sws_scale(
      swsContext_.get(),
      cpuFrame->data,
      cpuFrame->linesize,
      0,
      height,
      nv12CpuFrame->data,
      nv12CpuFrame->linesize);
  TORCH_CHECK(
      convertedHeight == height, "sws_scale failed for CPU->NV12 conversion");

  int ySize = width * height;
  TORCH_CHECK(
      ySize % 2 == 0,
      "Y plane size must be even. Please report on TorchCodec repo.");
  int uvSize = ySize / 2; // NV12: UV plane is half the size of Y plane
  size_t totalSize = static_cast<size_t>(ySize + uvSize);

  uint8_t* cudaBuffer = nullptr;
  cudaError_t err =
      cudaMalloc(reinterpret_cast<void**>(&cudaBuffer), totalSize);
  TORCH_CHECK(
      err == cudaSuccess,
      "Failed to allocate CUDA memory: ",
      cudaGetErrorString(err));

  UniqueAVFrame gpuFrame(av_frame_alloc());
  TORCH_CHECK(gpuFrame != nullptr, "Failed to allocate GPU AVFrame");

  gpuFrame->format = AV_PIX_FMT_CUDA;
  gpuFrame->width = width;
  gpuFrame->height = height;
  gpuFrame->data[0] = cudaBuffer;
  gpuFrame->data[1] = cudaBuffer + ySize;
  gpuFrame->linesize[0] = width;
  gpuFrame->linesize[1] = width;

  // Note that we use cudaMemcpy2D here instead of cudaMemcpy because the
  // linesizes (strides) may be different than the widths for the input CPU
  // frame. That's precisely what cudaMemcpy2D is for.
  err = cudaMemcpy2D(
      gpuFrame->data[0],
      gpuFrame->linesize[0],
      nv12CpuFrame->data[0],
      nv12CpuFrame->linesize[0],
      width,
      height,
      cudaMemcpyHostToDevice);
  TORCH_CHECK(
      err == cudaSuccess,
      "Failed to copy Y plane to GPU: ",
      cudaGetErrorString(err));

  TORCH_CHECK(
      height % 2 == 0,
      "height must be even. Please report on TorchCodec repo.");
  err = cudaMemcpy2D(
      gpuFrame->data[1],
      gpuFrame->linesize[1],
      nv12CpuFrame->data[1],
      nv12CpuFrame->linesize[1],
      width,
      height / 2,
      cudaMemcpyHostToDevice);
  TORCH_CHECK(
      err == cudaSuccess,
      "Failed to copy UV plane to GPU: ",
      cudaGetErrorString(err));

  ret = av_frame_copy_props(gpuFrame.get(), cpuFrame.get());
  TORCH_CHECK(
      ret >= 0,
      "Failed to copy frame properties: ",
      getFFMPEGErrorStringFromErrorCode(ret));

  // We're almost done, but we need to make sure the CUDA memory is freed
  // properly. Usually, AVFrame data is freed when av_frame_free() is called
  // (upon UniqueAVFrame destruction), but since we allocated the CUDA memory
  // ourselves, FFmpeg doesn't know how to free it. The recommended way to deal
  // with this is to associate the opaque_ref field of the AVFrame with a `free`
  // callback that will then be called by av_frame_free().
  gpuFrame->opaque_ref = av_buffer_create(
      nullptr, // data - we don't need any
      0, // data size
      cudaBufferFreeCallback, // callback triggered by av_frame_free()
      cudaBuffer, // parameter to callback
      0); // flags
  TORCH_CHECK(
      gpuFrame->opaque_ref != nullptr,
      "Failed to create GPU memory cleanup reference");

  return gpuFrame;
}

void BetaCudaDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  UniqueAVFrame gpuFrame =
      cpuFallback_ ? transferCpuFrameToGpuNV12(avFrame) : std::move(avFrame);

  // TODONVDEC P2: we may need to handle 10bit videos the same way the CUDA
  // ffmpeg interface does it with maybeConvertAVFrameToNV12OrRGB24().
  TORCH_CHECK(
      gpuFrame->format == AV_PIX_FMT_CUDA,
      "Expected CUDA format frame from BETA CUDA interface");

  validatePreAllocatedTensorShape(preAllocatedOutputTensor, gpuFrame);

  at::cuda::CUDAStream nvdecStream =
      at::cuda::getCurrentCUDAStream(device_.index());

  frameOutput.data = convertNV12FrameToRGB(
      gpuFrame, device_, nppCtx_, nvdecStream, preAllocatedOutputTensor);
}

std::string BetaCudaDeviceInterface::getDetails() {
  std::string details = "Beta CUDA Device Interface.";
  if (cpuFallback_) {
    details += " Using CPU fallback.";
    if (!nvcuvidAvailable_) {
      details += " NVCUVID not available!";
    }
  } else {
    details += " Using NVDEC.";
  }
  return details;
}

} // namespace facebook::torchcodec
