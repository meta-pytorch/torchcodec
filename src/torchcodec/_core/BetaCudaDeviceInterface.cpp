// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/types.h>
#include <mutex>
#include <vector>

#include "src/torchcodec/_core/BetaCudaDeviceInterface.h"

#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/NVDECCache.h"

// #include <cuda_runtime.h> // For cudaStreamSynchronize
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
  // Check decoder capabilities - same checks as DALI
  auto caps = CUVIDDECODECAPS{};
  caps.eCodecType = videoFormat->codec;
  caps.eChromaFormat = videoFormat->chroma_format;
  caps.nBitDepthMinus8 = videoFormat->bit_depth_luma_minus8;
  CUresult result = cuvidGetDecoderCaps(&caps);
  TORCH_CHECK(result == CUDA_SUCCESS, "Failed to get decoder caps: ", result);

  TORCH_CHECK(
      caps.bIsSupported,
      "Codec configuration not supported on this GPU. "
      "Codec: ",
      static_cast<int>(videoFormat->codec),
      ", chroma format: ",
      static_cast<int>(videoFormat->chroma_format),
      ", bit depth: ",
      videoFormat->bit_depth_luma_minus8 + 8);

  TORCH_CHECK(
      videoFormat->coded_width >= caps.nMinWidth &&
          videoFormat->coded_height >= caps.nMinHeight,
      "Video is too small in at least one dimension. Provided: ",
      videoFormat->coded_width,
      "x",
      videoFormat->coded_height,
      " vs supported:",
      caps.nMinWidth,
      "x",
      caps.nMinHeight);

  TORCH_CHECK(
      videoFormat->coded_width <= caps.nMaxWidth &&
          videoFormat->coded_height <= caps.nMaxHeight,
      "Video is too large in at least one dimension. Provided: ",
      videoFormat->coded_width,
      "x",
      videoFormat->coded_height,
      " vs supported:",
      caps.nMaxWidth,
      "x",
      caps.nMaxHeight);

  // See nMaxMBCount in cuviddec.h
  constexpr unsigned int macroblockConstant = 256;
  TORCH_CHECK(
      videoFormat->coded_width * videoFormat->coded_height /
              macroblockConstant <=
          caps.nMaxMBCount,
      "Video is too large (too many macroblocks). "
      "Provided (width * height / ",
      macroblockConstant,
      "): ",
      videoFormat->coded_width * videoFormat->coded_height / macroblockConstant,
      " vs supported:",
      caps.nMaxMBCount);

  // Decoder creation parameters, most are taken from DALI
  CUVIDDECODECREATEINFO decoderParams = {};
  decoderParams.bitDepthMinus8 = videoFormat->bit_depth_luma_minus8;
  decoderParams.ChromaFormat = videoFormat->chroma_format;
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
  result = cuvidCreateDecoder(decoder, &decoderParams);
  TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create NVDEC decoder: ", result);
  return UniqueCUvideodecoder(decoder, CUvideoDecoderDeleter{});
}

cudaVideoCodec validateCodecSupport(AVCodecID codecId) {
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
    default: {
      TORCH_CHECK(false, "Unsupported codec type: ", avcodec_get_name(codecId));
    }
  }
}

} // namespace

BetaCudaDeviceInterface::BetaCudaDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_cuda_beta, "BetaCudaDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());
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
    NVDECCache::getCache(device_.index())
        .returnDecoder(&videoFormat_, std::move(decoder_));
  }

  if (videoParser_) {
    // TODONVDEC P2: consider caching this? Does DALI do that?
    cuvidDestroyVideoParser(videoParser_);
    videoParser_ = nullptr;
  }
}

void BetaCudaDeviceInterface::initialize(
    const AVStream* avStream,
    const UniqueDecodingAVFormatContext& avFormatCtx) {
  torch::Tensor dummyTensorForCudaInitialization = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  auto cudaDevice = torch::Device(torch::kCUDA);
  defaultCudaInterface_ =
      std::unique_ptr<DeviceInterface>(createDeviceInterface(cudaDevice));
  AVCodecContext dummyCodecContext = {};
  defaultCudaInterface_->initialize(avStream, avFormatCtx);
  defaultCudaInterface_->registerHardwareDeviceWithCodec(&dummyCodecContext);

  TORCH_CHECK(avStream != nullptr, "AVStream cannot be null");
  timeBase_ = avStream->time_base;
  frameRateAvgFromFFmpeg_ = avStream->r_frame_rate;

  const AVCodecParameters* codecPar = avStream->codecpar;
  TORCH_CHECK(codecPar != nullptr, "CodecParameters cannot be null");

  initializeBSF(codecPar, avFormatCtx);

  // Create parser. Default values that aren't obvious are taken from DALI.
  CUVIDPARSERPARAMS parserParams = {};
  parserParams.CodecType = validateCodecSupport(codecPar->codec_id);
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
    decoder_ = NVDECCache::getCache(device_.index()).getDecoder(videoFormat);

    if (!decoder_) {
      // TODONVDEC P2: consider re-configuring an existing decoder instead of
      // re-creating one. See docs, see DALI.
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
  if (readyFrames_.empty()) {
    // No frame found, instruct caller to try again later after sending more
    // packets, or to stop if EOF was already sent.
    return eofSent_ ? AVERROR_EOF : AVERROR(EAGAIN);
  }

  CUVIDPARSERDISPINFO dispInfo = readyFrames_.front();
  readyFrames_.pop();

  // TODONVDEC P1 we need to set the procParams.output_stream field to the
  // current CUDA stream and ensure proper synchronization. There's a related
  // NVDECTODO in CudaDeviceInterface.cpp where we do the necessary
  // synchronization for NPP.
  CUVIDPROCPARAMS procParams = {};
  procParams.progressive_frame = dispInfo.progressive_frame;
  procParams.top_field_first = dispInfo.top_field_first;
  procParams.unpaired_field = dispInfo.repeat_first_field < 0;
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

void BetaCudaDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  TORCH_CHECK(
      avFrame->format == AV_PIX_FMT_CUDA,
      "Expected CUDA format frame from BETA CUDA interface");

  // TODONVDEC P1: we use the 'default' cuda device interface for color
  // conversion. That's a temporary hack to make things work. we should abstract
  // the color conversion stuff separately.
  defaultCudaInterface_->convertAVFrameToFrameOutput(
      avFrame, frameOutput, preAllocatedOutputTensor);
}

} // namespace facebook::torchcodec
