// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "GpuEncoder.h"

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include "CUDACommon.h"
#include "FFMPEGCommon.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

// Redefinition from CudaDeviceInterface.cpp anonymous namespace
int getFlagsAVHardwareDeviceContextCreate() {
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 26, 100)
  return AV_CUDA_USE_CURRENT_CONTEXT;
#else
  return 0;
#endif
}

// Redefinition from CudaDeviceInterface.cpp anonymous namespace
// TODO-VideoEncoder: unify device context creation, add caching to encoder
UniqueAVBufferRef createHardwareDeviceContext(const torch::Device& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find cuda device");

  int deviceIndex = getDeviceIndex(device);

  c10::cuda::CUDAGuard deviceGuard(device);
  // We set the device because we may be called from a different thread than
  // the one that initialized the cuda context.
  TORCH_CHECK(
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
    TORCH_CHECK(
        false,
        "Failed to create specified HW device. This typically happens when ",
        "your installed FFmpeg doesn't support CUDA (see ",
        "https://github.com/pytorch/torchcodec#installing-cuda-enabled-torchcodec",
        "). FFmpeg error: ", getFFMPEGErrorStringFromErrorCode(err));
    /* clang-format on */
  }

  return UniqueAVBufferRef(hardwareDeviceCtxRaw);
}

// RGB to NV12 color conversion matrices (inverse of YUV to RGB)
// Note: NPP's ColorTwist function apparently expects "limited range"
// coefficient format even when producing full range output. All matrices below
// use the limited range coefficient format (Y with +16 offset) for NPP
// compatibility.

// BT.601 limited range (matches FFmpeg default behavior)
const Npp32f defaultLimitedRangeRgbToNv12[3][4] = {
    // Y = 16 + 0.859 * (0.299*R + 0.587*G + 0.114*B)
    {0.257f, 0.504f, 0.098f, 16.0f},
    // U = -0.148*R - 0.291*G + 0.439*B + 128 (BT.601 coefficients)
    {-0.148f, -0.291f, 0.439f, 128.0f},
    // V = 0.439*R - 0.368*G - 0.071*B + 128 (BT.601 coefficients)
    {0.439f, -0.368f, -0.071f, 128.0f}};
} // anonymous namespace

GpuEncoder::GpuEncoder(const torch::Device& device) : device_(device) {
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());

  initializeCudaContextWithPytorch(device_);
  initializeHardwareContext();
}

GpuEncoder::~GpuEncoder() {}

void GpuEncoder::initializeHardwareContext() {
  hardwareDeviceCtx_ = createHardwareDeviceContext(device_);
  nppCtx_ = getNppStreamContext(device_);
}

std::optional<const AVCodec*> GpuEncoder::findEncoder(
    const AVCodecID& codecId) {
  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    if (codec->id != codecId || !av_codec_is_encoder(codec)) {
      continue;
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

void GpuEncoder::registerHardwareDeviceWithCodec(AVCodecContext* codecContext) {
  TORCH_CHECK(
      hardwareDeviceCtx_, "Hardware device context has not been initialized");
  TORCH_CHECK(codecContext != nullptr, "codecContext is null");
  codecContext->hw_device_ctx = av_buffer_ref(hardwareDeviceCtx_.get());
}

void GpuEncoder::setupEncodingContext(AVCodecContext* codecContext) {
  TORCH_CHECK(
      hardwareDeviceCtx_, "Hardware device context has not been initialized");
  TORCH_CHECK(codecContext != nullptr, "codecContext is null");

  codecContext->sw_pix_fmt = AV_PIX_FMT_NV12;
  codecContext->pix_fmt = AV_PIX_FMT_CUDA;

  AVBufferRef* hwFramesCtxRef = av_hwframe_ctx_alloc(hardwareDeviceCtx_.get());
  TORCH_CHECK(
      hwFramesCtxRef != nullptr,
      "Failed to allocate hardware frames context for codec");

  AVHWFramesContext* hwFramesCtx =
      reinterpret_cast<AVHWFramesContext*>(hwFramesCtxRef->data);
  hwFramesCtx->format = codecContext->pix_fmt;
  hwFramesCtx->sw_format = codecContext->sw_pix_fmt;
  hwFramesCtx->width = codecContext->width;
  hwFramesCtx->height = codecContext->height;

  int ret = av_hwframe_ctx_init(hwFramesCtxRef);
  if (ret < 0) {
    av_buffer_unref(&hwFramesCtxRef);
    TORCH_CHECK(
        false,
        "Failed to initialize CUDA frames context for codec: ",
        getFFMPEGErrorStringFromErrorCode(ret));
  }

  codecContext->hw_frames_ctx = hwFramesCtxRef;
}

UniqueAVFrame GpuEncoder::convertTensorToAVFrame(
    const torch::Tensor& tensor,
    [[maybe_unused]] AVPixelFormat targetFormat,
    int frameIndex,
    AVCodecContext* codecContext) {
  TORCH_CHECK(tensor.is_cuda(), "GpuEncoder requires CUDA tensors");
  TORCH_CHECK(
      tensor.dim() == 3 && tensor.size(0) == 3,
      "Expected 3D RGB tensor (CHW format), got shape: ",
      tensor.sizes());
  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame != nullptr, "Failed to allocate AVFrame");

  avFrame->format = AV_PIX_FMT_CUDA;
  avFrame->width = static_cast<int>(tensor.size(2));
  avFrame->height = static_cast<int>(tensor.size(1));
  avFrame->pts = frameIndex;

  int ret = av_hwframe_get_buffer(
      codecContext ? codecContext->hw_frames_ctx : nullptr, avFrame.get(), 0);
  TORCH_CHECK(
      ret >= 0,
      "Failed to allocate hardware frame: ",
      getFFMPEGErrorStringFromErrorCode(ret));

  // Validate that avFrame was properly allocated with CUDA memory
  TORCH_CHECK(
      avFrame != nullptr && avFrame->data[0] != nullptr,
      "avFrame must be pre-allocated with CUDA memory");

  // Convert CHW to HWC for NPP processing
  int height = static_cast<int>(tensor.size(1));
  int width = static_cast<int>(tensor.size(2));
  torch::Tensor hwcFrame = tensor.permute({1, 2, 0}).contiguous();

  // Get current CUDA stream for NPP operations
  at::cuda::CUDAStream currentStream =
      at::cuda::getCurrentCUDAStream(device_.index());

  // Setup NPP context with current stream
  nppCtx_->hStream = currentStream.stream();
  cudaError_t cudaErr =
      cudaStreamGetFlags(nppCtx_->hStream, &nppCtx_->nStreamFlags);
  TORCH_CHECK(
      cudaErr == cudaSuccess,
      "cudaStreamGetFlags failed: ",
      cudaGetErrorString(cudaErr));

  // Always use FFmpeg's default behavior: BT.601 limited range
  NppiSize oSizeROI = {width, height};

  NppStatus status = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(
      static_cast<const Npp8u*>(hwcFrame.data_ptr()),
      hwcFrame.stride(0) * hwcFrame.element_size(),
      avFrame->data,
      avFrame->linesize,
      oSizeROI,
      defaultLimitedRangeRgbToNv12,
      *nppCtx_);

  TORCH_CHECK(
      status == NPP_SUCCESS,
      "Failed to convert RGB to NV12: NPP error code ",
      status);

  // Validate CUDA operations completed successfully
  cudaError_t memCheck = cudaGetLastError();
  TORCH_CHECK(
      memCheck == cudaSuccess,
      "CUDA error detected: ",
      cudaGetErrorString(memCheck));

  // TODO-VideoEncoder: Enable configuration of color properties, similar to
  // FFmpeg Set color properties to FFmpeg defaults
  avFrame->colorspace = AVCOL_SPC_SMPTE170M; // BT.601
  avFrame->color_range = AVCOL_RANGE_MPEG; // Limited range

  return avFrame;
}

} // namespace facebook::torchcodec
