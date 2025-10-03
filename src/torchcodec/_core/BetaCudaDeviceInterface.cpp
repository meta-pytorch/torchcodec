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
  BetaCudaDeviceInterface* decoder =
      static_cast<BetaCudaDeviceInterface*>(pUserData);
  return decoder->streamPropertyChange(videoFormat);
}

static int CUDAAPI
pfnDecodePictureCallback(void* pUserData, CUVIDPICPARAMS* picParams) {
  BetaCudaDeviceInterface* decoder =
      static_cast<BetaCudaDeviceInterface*>(pUserData);
  return decoder->frameReadyForDecoding(picParams);
}

static int CUDAAPI
pfnDisplayPictureCallback(void* pUserData, CUVIDPARSERDISPINFO* dispInfo) {
  BetaCudaDeviceInterface* decoder =
      static_cast<BetaCudaDeviceInterface*>(pUserData);
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

  // Decoder creation parameters, taken from DALI
  CUVIDDECODECREATEINFO decoder_info = {};
  decoder_info.bitDepthMinus8 = videoFormat->bit_depth_luma_minus8;
  decoder_info.ChromaFormat = videoFormat->chroma_format;
  decoder_info.CodecType = videoFormat->codec;
  decoder_info.ulHeight = videoFormat->coded_height;
  decoder_info.ulWidth = videoFormat->coded_width;
  decoder_info.ulMaxHeight = videoFormat->coded_height;
  decoder_info.ulMaxWidth = videoFormat->coded_width;
  decoder_info.ulTargetHeight =
      videoFormat->display_area.bottom - videoFormat->display_area.top;
  decoder_info.ulTargetWidth =
      videoFormat->display_area.right - videoFormat->display_area.left;
  decoder_info.ulNumDecodeSurfaces = videoFormat->min_num_decode_surfaces;
  decoder_info.ulNumOutputSurfaces = 2;
  decoder_info.display_area.left = videoFormat->display_area.left;
  decoder_info.display_area.right = videoFormat->display_area.right;
  decoder_info.display_area.top = videoFormat->display_area.top;
  decoder_info.display_area.bottom = videoFormat->display_area.bottom;

  CUvideodecoder* decoder = new CUvideodecoder();
  result = cuvidCreateDecoder(decoder, &decoder_info);
  TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create NVDEC decoder: ", result);
  return UniqueCUvideodecoder(decoder, CUvideoDecoderDeleter{});
}

} // namespace

BetaCudaDeviceInterface::BetaCudaDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_cuda_beta, "BetaCudaDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());
}

BetaCudaDeviceInterface::~BetaCudaDeviceInterface() {
  // TODONVDEC P0: we probably need to free the frames that have been decoded by
  // NVDEC but not yet "mapped" - i.e. those that are still in readyFrames_?

  if (decoder_) {
    NVDECCache::getCache(device_.index())
        .returnDecoder(&videoFormat_, std::move(decoder_));
  }

  if (videoParser_) {
    // TODONVDEC P2: consider caching this? Does DALI do that?
    cuvidDestroyVideoParser(videoParser_);
    videoParser_ = nullptr;
  }
}

void BetaCudaDeviceInterface::initialize(const AVStream* avStream) {
  torch::Tensor dummyTensorForCudaInitialization = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  TORCH_CHECK(avStream != nullptr, "AVStream cannot be null");
  timeBase_ = avStream->time_base;

  auto cudaDevice = torch::Device(torch::kCUDA);
  defaultCudaInterface_ =
      std::unique_ptr<DeviceInterface>(createDeviceInterface(cudaDevice));
  AVCodecContext dummyCodecContext = {};
  defaultCudaInterface_->initialize(avStream);
  defaultCudaInterface_->registerHardwareDeviceWithCodec(&dummyCodecContext);

  const AVCodecParameters* codecpar = avStream->codecpar;
  TORCH_CHECK(codecpar != nullptr, "CodecParameters cannot be null");

  TORCH_CHECK(
      // TODONVDEC P0 support more
      avStream->codecpar->codec_id == AV_CODEC_ID_H264,
      "Can only do H264 for now");

  // Setup bit stream filters (BSF):
  // https://ffmpeg.org/doxygen/7.0/group__lavc__bsf.html
  // This is only needed for some formats, like H264 or HEVC.  TODONVDEC P1: For
  // now we apply BSF unconditionally, but it should be optional  and dependent
  // on codec and container.
  const AVBitStreamFilter* avBSF = av_bsf_get_by_name("h264_mp4toannexb");
  TORCH_CHECK(
      avBSF != nullptr, "Failed to find h264_mp4toannexb bitstream filter");

  AVBSFContext* avBSFContext = nullptr;
  int retVal = av_bsf_alloc(avBSF, &avBSFContext);
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to allocate bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  bitstreamFilter_.reset(avBSFContext);

  retVal = avcodec_parameters_copy(bitstreamFilter_->par_in, codecpar);
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to copy codec parameters: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  retVal = av_bsf_init(bitstreamFilter_.get());
  TORCH_CHECK(
      retVal == AVSUCCESS,
      "Failed to initialize bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  // Create parser. Default values that aren't obvious are taken from DALI.
  CUVIDPARSERPARAMS parserParams = {};
  parserParams.CodecType = cudaVideoCodec_H264;
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
      // TODONVDEC P0: consider re-configuring an existing decoder instead of
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
  CUVIDSOURCEDATAPACKET cuvidPacket = {};

  if (packet.get() && packet->data && packet->size > 0) {
    applyBSF(packet);

    // Regular packet with data
    cuvidPacket.payload = packet->data;
    cuvidPacket.payload_size = packet->size;
    cuvidPacket.flags = CUVID_PKT_TIMESTAMP;
    cuvidPacket.timestamp = packet->pts;

  } else {
    // End of stream packet
    cuvidPacket.flags = CUVID_PKT_ENDOFSTREAM;
    eofSent_ = true;
  }

  CUresult result = cuvidParseVideoData(videoParser_, &cuvidPacket);
  if (result != CUDA_SUCCESS) {
    return AVERROR_EXTERNAL;
  }
  return AVSUCCESS;
}

void BetaCudaDeviceInterface::applyBSF(ReferenceAVPacket& packet) {
  if (!bitstreamFilter_) {
    return;
  }

  int retVal = av_bsf_send_packet(bitstreamFilter_.get(), packet.get());
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to send packet to bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  // Create a temporary packet to receive the filtered data
  // TODO P1: the docs mention there can theoretically be multiple output
  // packets for a single input, i.e. we may need to call av_bsf_receive_packet
  // more than once. We should figure out whether that applies to the BSF we're
  // using.
  AutoAVPacket filteredAutoPacket;
  ReferenceAVPacket filteredPacket(filteredAutoPacket);
  retVal = av_bsf_receive_packet(bitstreamFilter_.get(), filteredPacket.get());
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to receive packet from bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  // Free the original packet's data which isn't needed anymore, and move the
  // fields of the filtered packet into the original packet. The filtered packet
  // fields are re-set by av_packet_move_ref, so when it goes out of scope and
  // gets destructed, it's not going to affect the original packet.
  packet.reset(filteredPacket);
  // TODONVDEC P0: consider cleaner ways to do this. Maybe we should let
  // applyBSF return a new packet, and maybe that new packet needs to be a field
  // on the interface to avoid complex lifetime issues.
}

// Parser triggers this callback within cuvidParseVideoData when a frame is
// ready to be decoded, i.e. the parser received all the necessary packets for a
// given frame. It means we can send that frame to be decoded by the hardware
// NVDEC decoder by calling cuvidDecodePicture which is non-blocking.
int BetaCudaDeviceInterface::frameReadyForDecoding(CUVIDPICPARAMS* picParams) {
  if (isFlushing_) {
    return 0;
  }

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
    // packets.
    return AVERROR(EAGAIN);
  }
  CUVIDPARSERDISPINFO dispInfo = readyFrames_.front();
  readyFrames_.pop();

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
  CUresult result = cuvidMapVideoFrame(
      *decoder_.get(), dispInfo.picture_index, &framePtr, &pitch, &procParams);

  if (result != CUDA_SUCCESS) {
    return AVERROR_EXTERNAL;
  }

  avFrame = convertCudaFrameToAVFrame(framePtr, pitch, dispInfo);

  // Unmap the frame so that the decoder can reuse its corresponding output
  // surface. Whether this is blocking is unclear?
  cuvidUnmapVideoFrame(*decoder_.get(), framePtr);
  // TODONVDEC P0: Get clarity on this:
  // We assume that the framePtr is still valid after unmapping. That framePtr
  // is now part of the avFrame, which we'll return to the caller, and the
  // caller will immediately use it for color-conversion, at which point a copy
  // happens. After the copy, it doesn't matter whether framePtr is still valid.
  // And we'll return to this function (and to cuvidUnmapVideoFrame()) *after*
  // the copy is made, so there should be no risk of overwriting the data before
  // the copy.
  // Buuuut yeah, we need get more clarity on what actually happens, and on
  // what's needed. IIUC DALI makes the color-conversion copy immediately after
  // cuvidMapVideoFrame() and *before* cuvidUnmapVideoFrame() with a synchronize
  // in between. So maybe we should do the same.

  return AVSUCCESS;
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

  // TODONVDEC P0: Zero division error!!!
  // TODONVDEC P0: Move AVRational arithmetic to FFMPEGCommon, and put the
  // similar SingleStreamDecoder stuff there too.
  unsigned int frameRateNum = videoFormat_.frame_rate.numerator;
  unsigned int frameRateDen = videoFormat_.frame_rate.denominator;
  int64_t duration = static_cast<int64_t>((frameRateDen * timeBase_.den)) /
      (frameRateNum * timeBase_.num);
  setDuration(avFrame, duration);

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
  isFlushing_ = true;

  // TODONVDEC P0: simplify flushing and "eofSent_" logic. We should just have a
  // "sendEofPacket()" function that does the right thing, instead of setting
  // CUVID_PKT_ENDOFSTREAM in different places.
  if (!eofSent_) {
    CUVIDSOURCEDATAPACKET cuvidPacket = {};
    cuvidPacket.flags = CUVID_PKT_ENDOFSTREAM;
    CUresult result = cuvidParseVideoData(videoParser_, &cuvidPacket);
    if (result == CUDA_SUCCESS) {
      eofSent_ = true;
    }
  }

  isFlushing_ = false;

  std::queue<CUVIDPARSERDISPINFO> emptyQueue;
  std::swap(readyFrames_, emptyQueue);

  eofSent_ = false;
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
