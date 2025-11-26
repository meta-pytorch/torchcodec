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

  return convertRGBTensorToNV12Frame(tensor, frameIndex, codecContext);
}

UniqueAVFrame GpuEncoder::convertRGBTensorToNV12Frame(
    const torch::Tensor& tensor,
    int frameIndex,
    AVCodecContext* codecContext) {
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

  at::cuda::CUDAStream currentStream =
      at::cuda::getCurrentCUDAStream(device_.index());

  facebook::torchcodec::convertRGBTensorToNV12Frame(
      tensor, avFrame, device_, nppCtx_, currentStream);

  // Set color properties to FFmpeg defaults
  avFrame->colorspace = AVCOL_SPC_SMPTE170M; // BT.601
  avFrame->color_range = AVCOL_RANGE_MPEG; // Limited range

  return avFrame;
}

} // namespace facebook::torchcodec
