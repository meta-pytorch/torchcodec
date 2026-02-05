// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cuda_runtime.h>
#include <mutex>

#include "Cache.h"
#include "CudaDeviceInterface.h"
#include "FFMPEGCommon.h"
#include "ValidationUtils.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

static bool g_cuda = registerDeviceInterface(
    DeviceInterfaceKey(kStableCUDA),
    [](const StableDevice& device) { return new CudaDeviceInterface(device); });

// We reuse cuda contexts across VideoDeoder instances. This is because
// creating a cuda context is expensive. The cache mechanism is as follows:
// 1. There is a cache of size MAX_CONTEXTS_PER_GPU_IN_CACHE cuda contexts for
//    each GPU.
// 2. When we destroy a SingleStreamDecoder instance we release the cuda context
// to
//    the cache if the cache is not full.
// 3. When we create a SingleStreamDecoder instance we try to get a cuda context
// from
//    the cache. If the cache is empty we create a new cuda context.

// Set to -1 to have an infinitely sized cache. Set it to 0 to disable caching.
// Set to a positive number to have a cache of that size.
const int MAX_CONTEXTS_PER_GPU_IN_CACHE = -1;
PerGpuCache<AVBufferRef, Deleterp<AVBufferRef, void, av_buffer_unref>>
    g_cached_hw_device_ctxs(MAX_CUDA_GPUS, MAX_CONTEXTS_PER_GPU_IN_CACHE);

int getFlagsAVHardwareDeviceContextCreate() {
// 58.26.100 introduced the concept of reusing the existing cuda context
// which is much faster and lower memory than creating a new cuda context.
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 26, 100)
  return AV_CUDA_USE_CURRENT_CONTEXT;
#else
  return 0;
#endif
}

UniqueAVBufferRef getHardwareDeviceContext(const StableDevice& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
  STABLE_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find cuda device");
  int deviceIndex = getDeviceIndex(device);

  UniqueAVBufferRef hardwareDeviceCtx = g_cached_hw_device_ctxs.get(device);
  if (hardwareDeviceCtx) {
    return hardwareDeviceCtx;
  }

  // Create hardware device context
  StableDeviceGuard deviceGuard(device.index());
  // We set the device because we may be called from a different thread than
  // the one that initialized the cuda context.
  STABLE_CHECK(
      cudaSetDevice(deviceIndex) == cudaSuccess, "Failed to set CUDA device");
  AVBufferRef* hardwareDeviceCtxRaw = nullptr;
  std::string deviceOrdinal = std::to_string(deviceIndex);

  int err = av_hwdevice_ctx_create(
      &hardwareDeviceCtxRaw,
      type,
      deviceOrdinal.c_str(),
      nullptr,
      getFlagsAVHardwareDeviceContextCreate());

  if (err < 0) {
    /* clang-format off */
    STABLE_CHECK(
        false,
        "Failed to create specified HW device. This typically happens when "
        "your installed FFmpeg doesn't support CUDA (see "
        "https://github.com/pytorch/torchcodec#installing-cuda-enabled-torchcodec"
        "). FFmpeg error: " + getFFMPEGErrorStringFromErrorCode(err));
    /* clang-format on */
  }

  return UniqueAVBufferRef(hardwareDeviceCtxRaw);
}

} // namespace

CudaDeviceInterface::CudaDeviceInterface(const StableDevice& device)
    : DeviceInterface(device) {
  STABLE_CHECK(g_cuda, "CudaDeviceInterface was not registered!");
  STABLE_CHECK(
      device_.type() == kStableCUDA, "Unsupported device: must be CUDA");

  initializeCudaContextWithPytorch(device_);

  // Resolve the device index early so we don't need to call cudaGetDevice()
  // at destruction time (when CUDA may already be shut down).
  if (!device_.has_index()) {
    int resolvedIndex = getDeviceIndex(device_);
    device_.set_index(static_cast<StableDeviceIndex>(resolvedIndex));
  }

  hardwareDeviceCtx_ = getHardwareDeviceContext(device_);
  nppCtx_ = getNppStreamContext(device_);
}

CudaDeviceInterface::~CudaDeviceInterface() {
  if (hardwareDeviceCtx_) {
    g_cached_hw_device_ctxs.addIfCacheHasCapacity(
        device_, std::move(hardwareDeviceCtx_));
  }
  returnNppStreamContextToCache(device_, std::move(nppCtx_));
}

void CudaDeviceInterface::initialize(
    const AVStream* avStream,
    const UniqueDecodingAVFormatContext& avFormatCtx,
    const SharedAVCodecContext& codecContext) {
  STABLE_CHECK(avStream != nullptr, "avStream is null");
  codecContext_ = codecContext;
  timeBase_ = avStream->time_base;

  // TODO: Ideally, we should keep all interface implementations independent.
  cpuInterface_ = createDeviceInterface(kStableCPU);
  STABLE_CHECK(
      cpuInterface_ != nullptr, "Failed to create CPU device interface");
  cpuInterface_->initialize(avStream, avFormatCtx, codecContext);
  cpuInterface_->initializeVideo(
      VideoStreamOptions(),
      {},
      /*resizedOutputDims=*/std::nullopt);
}

void CudaDeviceInterface::initializeVideo(
    const VideoStreamOptions& videoStreamOptions,
    [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>& transforms,
    [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims) {
  videoStreamOptions_ = videoStreamOptions;
}

void CudaDeviceInterface::registerHardwareDeviceWithCodec(
    AVCodecContext* codecContext) {
  STABLE_CHECK(
      hardwareDeviceCtx_, "Hardware device context has not been initialized");
  STABLE_CHECK(codecContext != nullptr, "codecContext is null");
  codecContext->hw_device_ctx = av_buffer_ref(hardwareDeviceCtx_.get());
}

UniqueAVFrame CudaDeviceInterface::maybeConvertAVFrameToNV12OrRGB24(
    UniqueAVFrame& avFrame) {
  // We need FFmpeg filters to handle those conversion cases which are not
  // directly implemented in CUDA or CPU device interface (in case of a
  // fallback).

  // Input frame is on CPU, we will just pass it to CPU device interface, so
  // skipping filters context as CPU device interface will handle everything for
  // us.
  if (avFrame->format != AV_PIX_FMT_CUDA) {
    return std::move(avFrame);
  }

  auto hwFramesCtx =
      reinterpret_cast<AVHWFramesContext*>(avFrame->hw_frames_ctx->data);
  STABLE_CHECK(
      hwFramesCtx != nullptr,
      "The AVFrame does not have a hw_frames_ctx. "
      "That's unexpected, please report this to the TorchCodec repo.");

  AVPixelFormat actualFormat = hwFramesCtx->sw_format;

  // If the frame is already in NV12 format, we don't need to do anything.
  if (actualFormat == AV_PIX_FMT_NV12) {
    return std::move(avFrame);
  }

  AVPixelFormat outputFormat;
  std::stringstream filters;

  unsigned version_int = avfilter_version();
  if (version_int < AV_VERSION_INT(8, 0, 103)) {
    // Color conversion support ('format=' option) was added to scale_cuda from
    // n5.0. With the earlier version of ffmpeg we have no choice but use CPU
    // filters. See:
    // https://github.com/FFmpeg/FFmpeg/commit/62dc5df941f5e196164c151691e4274195523e95
    outputFormat = AV_PIX_FMT_RGB24;

    auto actualFormatName = av_get_pix_fmt_name(actualFormat);
    STABLE_CHECK(
        actualFormatName != nullptr,
        "The actual format of a frame is unknown to FFmpeg. "
        "That's unexpected, please report this to the TorchCodec repo.");

    filters << "hwdownload,format=" << actualFormatName;
  } else {
    // Actual output color format will be set via filter options
    outputFormat = AV_PIX_FMT_CUDA;

    filters << "scale_cuda=format=nv12:interp_algo=bilinear";
  }

  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);

  auto newContext = std::make_unique<FiltersContext>(
      avFrame->width,
      avFrame->height,
      frameFormat,
      avFrame->sample_aspect_ratio,
      avFrame->width,
      avFrame->height,
      outputFormat,
      filters.str(),
      timeBase_,
      av_buffer_ref(avFrame->hw_frames_ctx));

  if (!nv12Conversion_ || *nv12ConversionContext_ != *newContext) {
    nv12Conversion_ =
        std::make_unique<FilterGraph>(*newContext, videoStreamOptions_);
    nv12ConversionContext_ = std::move(newContext);
  }
  auto filteredAVFrame = nv12Conversion_->convert(avFrame);

  // If this check fails it means the frame wasn't
  // reshaped to its expected dimensions by filtergraph.
  STABLE_CHECK(
      (filteredAVFrame->width == nv12ConversionContext_->outputWidth) &&
          (filteredAVFrame->height == nv12ConversionContext_->outputHeight),
      "Expected frame from filter graph of " +
          std::to_string(nv12ConversionContext_->outputWidth) + "x" +
          std::to_string(nv12ConversionContext_->outputHeight) + ", got " +
          std::to_string(filteredAVFrame->width) + "x" +
          std::to_string(filteredAVFrame->height));

  return filteredAVFrame;
}

void CudaDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<StableTensor> preAllocatedOutputTensor) {
  validatePreAllocatedTensorShape(preAllocatedOutputTensor, avFrame);

  hasDecodedFrame_ = true;

  // All of our CUDA decoding assumes NV12 format. We handle non-NV12 formats by
  // converting them to NV12.
  avFrame = maybeConvertAVFrameToNV12OrRGB24(avFrame);

  if (avFrame->format != AV_PIX_FMT_CUDA) {
    // The frame's format is AV_PIX_FMT_CUDA if and only if its content is on
    // the GPU. In this branch, the frame is on the CPU. There are two possible
    // reasons:
    //
    //   1. During maybeConvertAVFrameToNV12OrRGB24(), we had a non-NV12 format
    //      frame and we're on FFmpeg 4.4 or earlier. In such cases, we had to
    //      use CPU filters and we just converted the frame to RGB24.
    //   2. This is what NVDEC gave us if it wasn't able to decode a frame, for
    //      whatever reason. Typically that happens if the video's encoder isn't
    //      supported by NVDEC.
    //
    // In both cases, we have a frame on the CPU. We send the frame back to the
    // CUDA device when we're done.

    enum AVPixelFormat frameFormat =
        static_cast<enum AVPixelFormat>(avFrame->format);

    FrameOutput cpuFrameOutput;
    if (frameFormat == AV_PIX_FMT_RGB24) {
      // Reason 1 above. The frame is already in RGB24, we just need to convert
      // it to a tensor.
      cpuFrameOutput.data = rgbAVFrameToTensor(avFrame);
    } else {
      // Reason 2 above. We need to do a full conversion which requires an
      // actual CPU device.
      cpuInterface_->convertAVFrameToFrameOutput(avFrame, cpuFrameOutput);
    }

    // Finally, we need to send the frame back to the GPU. Note that the
    // pre-allocated tensor is on the GPU, so we can't send that to the CPU
    // device interface. We copy it over here.
    if (preAllocatedOutputTensor.has_value()) {
      stableCopy_(preAllocatedOutputTensor.value(), cpuFrameOutput.data);
      frameOutput.data = preAllocatedOutputTensor.value();
    } else {
      frameOutput.data = stableTo(cpuFrameOutput.data, device_);
    }

    usingCPUFallback_ = true;
    return;
  }

  usingCPUFallback_ = false;

  // Above we checked that the AVFrame was on GPU, but that's not enough, we
  // also need to check that the AVFrame is in AV_PIX_FMT_NV12 format (8 bits),
  // because this is what the NPP color conversion routines expect. This SHOULD
  // be enforced by our call to maybeConvertAVFrameToNV12OrRGB24() above.
  STABLE_CHECK(
      avFrame->hw_frames_ctx != nullptr,
      "The AVFrame does not have a hw_frames_ctx. This should never happen");
  AVHWFramesContext* hwFramesCtx =
      reinterpret_cast<AVHWFramesContext*>(avFrame->hw_frames_ctx->data);
  STABLE_CHECK(
      hwFramesCtx != nullptr,
      "The AVFrame does not have a valid hw_frames_ctx. This should never happen");

  AVPixelFormat actualFormat = hwFramesCtx->sw_format;
  STABLE_CHECK(
      actualFormat == AV_PIX_FMT_NV12,
      std::string("The AVFrame is ") +
          (av_get_pix_fmt_name(actualFormat) ? av_get_pix_fmt_name(actualFormat)
                                             : "unknown") +
          ", but we expected AV_PIX_FMT_NV12. "
          "That's unexpected, please report this to the TorchCodec repo.");

  // Figure out the NVDEC stream from the avFrame's hardware context.
  // In reality, we know that this stream is hardcoded to be the default stream
  // by FFmpeg:
  // https://github.com/FFmpeg/FFmpeg/blob/66e40840d15b514f275ce3ce2a4bf72ec68c7311/libavutil/hwcontext_cuda.c#L387-L388
  STABLE_CHECK(
      hwFramesCtx->device_ctx != nullptr,
      "The AVFrame's hw_frames_ctx does not have a device_ctx. ");
  auto cudaDeviceCtx =
      static_cast<AVCUDADeviceContext*>(hwFramesCtx->device_ctx->hwctx);
  STABLE_CHECK(cudaDeviceCtx != nullptr, "The hardware context is null");
  cudaStream_t nvdecStream = cudaDeviceCtx->stream;

  frameOutput.data = convertNV12FrameToRGB(
      avFrame, device_, nppCtx_, nvdecStream, preAllocatedOutputTensor);
}

// inspired by https://github.com/FFmpeg/FFmpeg/commit/ad67ea9
// we have to do this because of an FFmpeg bug where hardware decoding is not
// appropriately set, so we just go off and find the matching codec for the CUDA
// device
std::optional<const AVCodec*> CudaDeviceInterface::findCodec(
    const AVCodecID& codecId,
    bool isDecoder) {
  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    STABLE_CHECK(
        codec != nullptr,
        "codec returned by av_codec_iterate should not be null");
    if (isDecoder) {
      if (codec->id != codecId || !av_codec_is_decoder(codec)) {
        continue;
      }
    } else {
      if (codec->id != codecId || !av_codec_is_encoder(codec)) {
        continue;
      }
    }

    const AVCodecHWConfig* config = nullptr;
    for (int j = 0; (config = avcodec_get_hw_config(codec, j)) != nullptr;
         ++j) {
      if (config->device_type == AV_HWDEVICE_TYPE_CUDA) {
        return codec;
      }
    }
  }

  return std::nullopt;
}

std::string CudaDeviceInterface::getDetails() {
  // Note: for this interface specifically the fallback is only known after a
  // frame has been decoded, not before: that's when FFmpeg decides to fallback,
  // so we can't know earlier.
  if (!hasDecodedFrame_) {
    return std::string(
        "FFmpeg CUDA Device Interface. Fallback status unknown (no frames decoded).");
  }
  return std::string("FFmpeg CUDA Device Interface. Using ") +
      (usingCPUFallback_ ? "CPU fallback." : "NVDEC.");
}

// --------------------------------------------------------------------------
// Below are methods exclusive to video encoding:
// --------------------------------------------------------------------------
namespace {
// Note: [RGB -> YUV Color Conversion, limited color range]
//
// For context on this subject, first read the note:
// [YUV -> RGB Color Conversion, color space and color range]
// https://github.com/meta-pytorch/torchcodec/blob/main/src/torchcodec/_core/CUDACommon.cpp#L63-L65
//
// Lets encode RGB -> YUV in the limited color range for BT.601 color space.
// In limited range, the [0, 255] range is mapped into [16-235] for Y, and into
// [16-240] for U,V.
// To implement, we get the full range conversion matrix as before, then scale:
// - Y channel: scale by (235-16)/255 = 219/255
// - U,V channels: scale by (240-16)/255 = 224/255
// https://en.wikipedia.org/wiki/YCbCr#Y%E2%80%B2PbPr_to_Y%E2%80%B2CbCr
//
// ```py
// import torch
// kr, kg, kb = 0.299, 0.587, 0.114  # BT.601 luma coefficients
// u_scale = 2 * (1 - kb)
// v_scale = 2 * (1 - kr)
//
// rgb_to_yuv_full = torch.tensor([
//     [kr, kg, kb],
//     [-kr/u_scale, -kg/u_scale, (1-kb)/u_scale],
//     [(1-kr)/v_scale, -kg/v_scale, -kb/v_scale]
// ])
//
// full_to_limited_y_scale = 219.0 / 255.0
// full_to_limited_uv_scale = 224.0 / 255.0
//
// rgb_to_yuv_limited = rgb_to_yuv_full * torch.tensor([
//     [full_to_limited_y_scale],
//     [full_to_limited_uv_scale],
//     [full_to_limited_uv_scale]
// ])
//
// print("RGB->YUV matrix (Limited Range BT.601):")
// print(rgb_to_yuv_limited)
// ```
//
// This yields:
// tensor([[ 0.2568,  0.5041,  0.0979],
//         [-0.1482, -0.2910,  0.4392],
//         [ 0.4392, -0.3678, -0.0714]])
//
// Which matches https://fourcc.org/fccyvrgb.php
//
// To perform color conversion in NPP, we are required to provide these color
// conversion matrices to ColorTwist functions, for example,
// `nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx`.
// https://docs.nvidia.com/cuda/npp/image_color_conversion.html
//
// These offsets are added in the 4th column of each conversion matrix below.
// - In limited range, Y is offset by 16 to add the lower margin.
// - In both color ranges, U,V are offset by 128 to be centered around 0.
//
// RGB to YUV conversion matrices to use in NPP color conversion functions
struct ColorConversionMatrices {
  static constexpr Npp32f BT601_LIMITED[3][4] = {
      {0.2568f, 0.5041f, 0.0979f, 16.0f},
      {-0.1482f, -0.2910f, 0.4392f, 128.0f},
      {0.4392f, -0.3678f, -0.0714f, 128.0f}};

  static constexpr Npp32f BT601_FULL[3][4] = {
      {0.2990f, 0.5870f, 0.1140f, 0.0f},
      {-0.1687f, -0.3313f, 0.5000f, 128.0f},
      {0.5000f, -0.4187f, -0.0813f, 128.0f}};

  static constexpr Npp32f BT709_LIMITED[3][4] = {
      {0.1826f, 0.6142f, 0.0620f, 16.0f},
      {-0.1006f, -0.3386f, 0.4392f, 128.0f},
      {0.4392f, -0.3989f, -0.0403f, 128.0f}};

  static constexpr Npp32f BT709_FULL[3][4] = {
      {0.2126f, 0.7152f, 0.0722f, 0.0f},
      {-0.1146f, -0.3854f, 0.5000f, 128.0f},
      {0.5000f, -0.4542f, -0.0458f, 128.0f}};

  static constexpr Npp32f BT2020_LIMITED[3][4] = {
      {0.2256f, 0.5823f, 0.0509f, 16.0f},
      {-0.1227f, -0.3166f, 0.4392f, 128.0f},
      {0.4392f, -0.4039f, -0.0353f, 128.0f}};

  static constexpr Npp32f BT2020_FULL[3][4] = {
      {0.2627f, 0.6780f, 0.0593f, 0.0f},
      {-0.139630f, -0.360370f, 0.5000f, 128.0f},
      {0.5000f, -0.459786f, -0.040214f, 128.0f}};
};

// Returns conversion matrix based on codec context color space and range
const Npp32f (*getConversionMatrix(AVCodecContext* codecContext))[4] {
  if (codecContext->color_range == AVCOL_RANGE_MPEG || // limited range
      codecContext->color_range == AVCOL_RANGE_UNSPECIFIED) {
    if (codecContext->colorspace == AVCOL_SPC_BT470BG) {
      return ColorConversionMatrices::BT601_LIMITED;
    } else if (codecContext->colorspace == AVCOL_SPC_BT709) {
      return ColorConversionMatrices::BT709_LIMITED;
    } else if (codecContext->colorspace == AVCOL_SPC_BT2020_NCL) {
      return ColorConversionMatrices::BT2020_LIMITED;
    } else { // default to BT.601
      return ColorConversionMatrices::BT601_LIMITED;
    }
  } else if (codecContext->color_range == AVCOL_RANGE_JPEG) { // full range
    if (codecContext->colorspace == AVCOL_SPC_BT470BG) {
      return ColorConversionMatrices::BT601_FULL;
    } else if (codecContext->colorspace == AVCOL_SPC_BT709) {
      return ColorConversionMatrices::BT709_FULL;
    } else if (codecContext->colorspace == AVCOL_SPC_BT2020_NCL) {
      return ColorConversionMatrices::BT2020_FULL;
    } else { // default to BT.601
      return ColorConversionMatrices::BT601_FULL;
    }
  }
  return ColorConversionMatrices::BT601_LIMITED;
}
} // namespace

UniqueAVFrame CudaDeviceInterface::convertTensorToAVFrameForEncoding(
    const StableTensor& tensor,
    int frameIndex,
    AVCodecContext* codecContext) {
  STABLE_CHECK(
      tensor.dim() == 3 && tensor.size(0) == 3,
      "Expected 3D RGB tensor (CHW format), got shape with dim=" +
          std::to_string(tensor.dim()));
  STABLE_CHECK(
      tensor.device().type() == kStableCUDA, "Expected tensor on CUDA device");

  UniqueAVFrame avFrame(av_frame_alloc());
  STABLE_CHECK(avFrame != nullptr, "Failed to allocate AVFrame");
  int height = static_cast<int>(tensor.size(1));
  int width = static_cast<int>(tensor.size(2));

  // TODO-VideoEncoder: (P1) Unify AVFrame creation with CPU method
  avFrame->format = AV_PIX_FMT_CUDA;
  avFrame->height = height;
  avFrame->width = width;
  avFrame->pts = frameIndex;

  // FFmpeg's av_hwframe_get_buffer is used to allocate memory on CUDA device.
  // TODO-VideoEncoder: (P2) Consider using pytorch to allocate CUDA memory for
  // efficiency
  int ret =
      av_hwframe_get_buffer(codecContext->hw_frames_ctx, avFrame.get(), 0);
  STABLE_CHECK(
      ret >= 0,
      "Failed to allocate hardware frame: " +
          getFFMPEGErrorStringFromErrorCode(ret));

  STABLE_CHECK(
      avFrame != nullptr && avFrame->data[0] != nullptr,
      "avFrame must be pre-allocated with CUDA memory");

  // TODO VideoEncoder: Investigate ways to avoid this copy
  StableTensor hwcFrame = stableContiguous(stablePermute(tensor, {1, 2, 0}));

  NppiSize oSizeROI = {width, height};
  NppStatus status;
  // Convert to NV12, as CUDA_ENCODING_PIXEL_FORMAT is always NV12 currently
  status = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(
      hwcFrame.const_data_ptr<Npp8u>(),
      validateInt64ToInt(
          hwcFrame.stride(0) * hwcFrame.element_size(), "nSrcStep"),
      avFrame->data,
      avFrame->linesize,
      oSizeROI,
      getConversionMatrix(codecContext),
      *nppCtx_);

  STABLE_CHECK(
      status == NPP_SUCCESS,
      std::string("Failed to convert RGB to ") +
          av_get_pix_fmt_name(DeviceInterface::CUDA_ENCODING_PIXEL_FORMAT) +
          ": NPP error code " + std::to_string(status));

  avFrame->colorspace = codecContext->colorspace;
  avFrame->color_range = codecContext->color_range;
  return avFrame;
}

// Allocates and initializes AVHWFramesContext, and sets pixel format fields
// to enable encoding with CUDA device. The hw_frames_ctx field is needed by
// FFmpeg to allocate frames on GPU's memory.
void CudaDeviceInterface::setupHardwareFrameContextForEncoding(
    AVCodecContext* codecContext) {
  STABLE_CHECK(codecContext != nullptr, "codecContext is null");
  STABLE_CHECK(
      hardwareDeviceCtx_, "Hardware device context has not been initialized");

  AVBufferRef* hwFramesCtxRef = av_hwframe_ctx_alloc(hardwareDeviceCtx_.get());
  STABLE_CHECK(
      hwFramesCtxRef != nullptr,
      "Failed to allocate hardware frames context for codec");

  codecContext->sw_pix_fmt = DeviceInterface::CUDA_ENCODING_PIXEL_FORMAT;
  // Always set pixel format to support CUDA encoding.
  codecContext->pix_fmt = AV_PIX_FMT_CUDA;

  AVHWFramesContext* hwFramesCtx =
      reinterpret_cast<AVHWFramesContext*>(hwFramesCtxRef->data);
  hwFramesCtx->format = codecContext->pix_fmt;
  hwFramesCtx->sw_format = codecContext->sw_pix_fmt;
  hwFramesCtx->width = codecContext->width;
  hwFramesCtx->height = codecContext->height;

  int ret = av_hwframe_ctx_init(hwFramesCtxRef);
  if (ret < 0) {
    av_buffer_unref(&hwFramesCtxRef);
    STABLE_CHECK(
        false,
        "Failed to initialize CUDA frames context for codec: " +
            getFFMPEGErrorStringFromErrorCode(ret));
  }
  codecContext->hw_frames_ctx = hwFramesCtxRef;
}
} // namespace facebook::torchcodec
