// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "EncodingCpuDeviceInterface.h"

extern "C" {
#include <libswscale/swscale.h>
}

namespace facebook::torchcodec {

namespace {
static bool g_encoding_cpu = registerEncodingDeviceInterface(
    EncodingDeviceInterfaceKey(torch::kCPU),
    [](const torch::Device& device) {
      return new EncodingCpuDeviceInterface(device);
    });
} // namespace

EncodingCpuDeviceInterface::EncodingCpuDeviceInterface(
    const torch::Device& device)
    : EncodingDeviceInterface(device) {
  TORCH_CHECK(g_encoding_cpu, "EncodingCpuDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCPU, "Unsupported device: ", device_.str());
}

void EncodingCpuDeviceInterface::initialize(AVCodecContext* codecContext) {
  TORCH_CHECK(codecContext != nullptr, "codecContext is null");

  outWidth_ = codecContext->width;
  outHeight_ = codecContext->height;
  outPixelFormat_ = codecContext->pix_fmt;
}

UniqueAVFrame EncodingCpuDeviceInterface::convertTensorToAVFrame(
    const torch::Tensor& frame,
    int frameIndex,
    [[maybe_unused]] AVCodecContext* codecContext) {
  // Extract dimensions from tensor (CHW format)
  TORCH_CHECK(
      frame.dim() == 3 && frame.size(0) == 3,
      "Expected 3D RGB tensor (CHW format), got shape: ",
      frame.sizes());

  inHeight_ = static_cast<int>(frame.size(1));
  inWidth_ = static_cast<int>(frame.size(2));

  // Initialize and cache scaling context if it does not exist
  if (!swsContext_) {
    swsContext_.reset(sws_getContext(
        inWidth_,
        inHeight_,
        inPixelFormat_,
        outWidth_,
        outHeight_,
        outPixelFormat_,
        SWS_BICUBIC, // Used by FFmpeg CLI
        nullptr,
        nullptr,
        nullptr));
    TORCH_CHECK(swsContext_ != nullptr, "Failed to create scaling context");
  }

  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame != nullptr, "Failed to allocate AVFrame");

  // Set output frame properties
  avFrame->format = outPixelFormat_;
  avFrame->width = outWidth_;
  avFrame->height = outHeight_;
  avFrame->pts = frameIndex;

  int status = av_frame_get_buffer(avFrame.get(), 0);
  TORCH_CHECK(status >= 0, "Failed to allocate frame buffer");

  // Need to convert/scale the frame
  // Create temporary frame with input format
  UniqueAVFrame inputFrame(av_frame_alloc());
  TORCH_CHECK(inputFrame != nullptr, "Failed to allocate input AVFrame");

  inputFrame->format = inPixelFormat_;
  inputFrame->width = inWidth_;
  inputFrame->height = inHeight_;

  uint8_t* tensorData = static_cast<uint8_t*>(frame.data_ptr());

  // TODO-VideoEncoder: Reorder tensor if in NHWC format
  int channelSize = inHeight_ * inWidth_;
  // Reorder RGB -> GBR for AV_PIX_FMT_GBRP format
  // TODO-VideoEncoder: Determine if FFmpeg supports planar RGB input format
  inputFrame->data[0] = tensorData + channelSize;
  inputFrame->data[1] = tensorData + (2 * channelSize);
  inputFrame->data[2] = tensorData;

  inputFrame->linesize[0] = inWidth_;
  inputFrame->linesize[1] = inWidth_;
  inputFrame->linesize[2] = inWidth_;

  status = sws_scale(
      swsContext_.get(),
      inputFrame->data,
      inputFrame->linesize,
      0,
      inputFrame->height,
      avFrame->data,
      avFrame->linesize);
  TORCH_CHECK(status == outHeight_, "sws_scale failed");

  return avFrame;
}

AVPixelFormat EncodingCpuDeviceInterface::getEncodingPixelFormat() const {
  return outPixelFormat_;
}

} // namespace facebook::torchcodec
