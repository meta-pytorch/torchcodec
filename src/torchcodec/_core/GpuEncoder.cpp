// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "GpuEncoder.h"
#include "ValidationUtils.h"

#include <c10/cuda/CUDAGuard.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
}

namespace facebook::torchcodec {

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
// https://en.wikipedia.org/wiki/YCbCr#Y%E2%80%99PbPr_to_Y%E2%80%99CbCr
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

GpuEncoder::GpuEncoder(const torch::Device& device) : device_(device) {
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());

  initializeCudaContextWithPytorch(device_);

  hardwareDeviceCtx_ = getHardwareDeviceContext(device_);
  nppCtx_ = getNppStreamContext(device_);
}

GpuEncoder::~GpuEncoder() {
  if (hardwareDeviceCtx_) {
    addHardwareDeviceContextToCache(device_, std::move(hardwareDeviceCtx_));
  }
  returnNppStreamContextToCache(device_, std::move(nppCtx_));
}

std::optional<const AVCodec*> GpuEncoder::findHardwareEncoder(
    const AVCodecID& codecId) {
  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    TORCH_CHECK(
        codec != nullptr,
        "codec returned by av_codec_iterate should not be null");
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

void GpuEncoder::setupHardwareFrameContextForEncoding(
    AVCodecContext* codecContext) {
  TORCH_CHECK(codecContext != nullptr, "codecContext is null");
  TORCH_CHECK(
      hardwareDeviceCtx_, "Hardware device context has not been initialized");

  AVBufferRef* hwFramesCtxRef = av_hwframe_ctx_alloc(hardwareDeviceCtx_.get());
  TORCH_CHECK(
      hwFramesCtxRef != nullptr,
      "Failed to allocate hardware frames context for codec");

  codecContext->sw_pix_fmt = CUDA_ENCODING_PIXEL_FORMAT;
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
    TORCH_CHECK(
        false,
        "Failed to initialize CUDA frames context for codec: ",
        getFFMPEGErrorStringFromErrorCode(ret));
  }
  codecContext->hw_frames_ctx = hwFramesCtxRef;
}

UniqueAVFrame GpuEncoder::convertCUDATensorToAVFrameForEncoding(
    const torch::Tensor& tensor,
    int frameIndex,
    AVCodecContext* codecContext) {
  TORCH_CHECK(
      tensor.dim() == 3 && tensor.size(0) == 3,
      "Expected 3D RGB tensor (CHW format), got shape: ",
      tensor.sizes());
  TORCH_CHECK(
      tensor.device().type() == torch::kCUDA,
      "Expected tensor on CUDA device, got: ",
      tensor.device().str());

  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame != nullptr, "Failed to allocate AVFrame");
  int height = static_cast<int>(tensor.size(1));
  int width = static_cast<int>(tensor.size(2));

  avFrame->format = AV_PIX_FMT_CUDA;
  avFrame->height = height;
  avFrame->width = width;
  avFrame->pts = frameIndex;

  // FFmpeg's av_hwframe_get_buffer is used to allocate memory on CUDA device.
  int ret =
      av_hwframe_get_buffer(codecContext->hw_frames_ctx, avFrame.get(), 0);
  TORCH_CHECK(
      ret >= 0,
      "Failed to allocate hardware frame: ",
      getFFMPEGErrorStringFromErrorCode(ret));

  TORCH_CHECK(
      avFrame != nullptr && avFrame->data[0] != nullptr,
      "avFrame must be pre-allocated with CUDA memory");

  torch::Tensor hwcFrame = tensor.permute({1, 2, 0}).contiguous();

  NppiSize oSizeROI = {width, height};
  NppStatus status;
  // Convert to NV12, as CUDA_ENCODING_PIXEL_FORMAT is always NV12 currently
  status = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(
      static_cast<const Npp8u*>(hwcFrame.data_ptr()),
      validateInt64ToInt(
          hwcFrame.stride(0) * hwcFrame.element_size(), "nSrcStep"),
      avFrame->data,
      avFrame->linesize,
      oSizeROI,
      getConversionMatrix(codecContext),
      *nppCtx_);

  TORCH_CHECK(
      status == NPP_SUCCESS,
      "Failed to convert RGB to ",
      av_get_pix_fmt_name(CUDA_ENCODING_PIXEL_FORMAT),
      ": NPP error code ",
      status);

  avFrame->colorspace = codecContext->colorspace;
  avFrame->color_range = codecContext->color_range;
  return avFrame;
}

std::unique_ptr<GpuEncoder> createGpuEncoder(const torch::Device& device) {
  return std::make_unique<GpuEncoder>(device);
}

} // namespace facebook::torchcodec
