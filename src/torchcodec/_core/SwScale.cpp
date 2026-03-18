// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "SwScale.h"
#include "Frame.h"

namespace facebook::torchcodec {

SwScale::SwScale(
    const SwsConfig& config,
    AVPixelFormat outputFormat,
    int swsFlags)
    : config_(config), outputFormat_(outputFormat), swsFlags_(swsFlags) {
  needsResize_ =
      (config_.inputHeight != config_.outputHeight ||
       config_.inputWidth != config_.outputWidth);

  bytesPerPixel_ = (outputFormat_ == AV_PIX_FMT_RGB48) ? 6 : 3;

  // Create color conversion context (input format -> output RGB format).
  // Color conversion always outputs at the input resolution.
  // When no resize is needed, input and output resolutions are the same.
  SwsConfig colorConversionFrameConfig(
      config_.inputWidth,
      config_.inputHeight,
      config_.inputFormat,
      config_.inputColorspace,
      config_.inputWidth,
      config_.inputHeight);

  colorConversionSwsContext_ = createSwsContext(
      colorConversionFrameConfig,
      // See [Transform and Format Conversion Order] for more on the output
      // pixel format.
      /*outputFormat=*/outputFormat_,
      // No flags for color conversion. When resizing is needed, we use a
      // separate swscale context with the appropriate resize flags.
      /*swsFlags=*/0);

  // Create resize context if needed (output RGB at input resolution ->
  // output RGB at output resolution).
  if (needsResize_) {
    SwsConfig resizeFrameConfig(
        config_.inputWidth,
        config_.inputHeight,
        outputFormat_,
        AVCOL_SPC_RGB,
        config_.outputWidth,
        config_.outputHeight);

    resizeSwsContext_ = createSwsContext(
        resizeFrameConfig,
        /*outputFormat=*/outputFormat_,
        /*swsFlags=*/swsFlags_);
  }
}

int SwScale::convert(
    const UniqueAVFrame& avFrame,
    torch::stable::Tensor& outputTensor) {
  // When resizing is needed, we do sws_scale twice: first convert to output
  // RGB at original resolution, then resize in output RGB space. This ensures
  // transforms happen in the output color space (RGB) rather than the input
  // color space (YUV).
  //
  // When no resize is needed, we do color conversion directly into the output
  // tensor.
  int inputBitDepth = (outputFormat_ == AV_PIX_FMT_RGB48) ? 16 : 8;
  torch::stable::Tensor colorConvertedTensor = needsResize_
      ? allocateEmptyHWCTensor(
            FrameDims(config_.inputHeight, config_.inputWidth, inputBitDepth),
            kStableCPU)
      : outputTensor;

  // sws_scale always takes uint8_t* pointers regardless of actual bit depth.
  uint8_t* colorConvertedPointers[4] = {
      static_cast<uint8_t*>(colorConvertedTensor.mutable_data_ptr()),
      nullptr,
      nullptr,
      nullptr};
  int colorConvertedWidth = static_cast<int>(colorConvertedTensor.sizes()[1]);
  int colorConvertedLinesizes[4] = {
      colorConvertedWidth * bytesPerPixel_, 0, 0, 0};

  int colorConvertedHeight = sws_scale(
      colorConversionSwsContext_.get(),
      avFrame->data,
      avFrame->linesize,
      0,
      avFrame->height,
      colorConvertedPointers,
      colorConvertedLinesizes);

  STD_TORCH_CHECK(
      colorConvertedHeight == avFrame->height,
      "Color conversion swscale pass failed: colorConvertedHeight != avFrame->height: ",
      colorConvertedHeight,
      " != ",
      avFrame->height);

  if (needsResize_) {
    uint8_t* srcPointers[4] = {
        static_cast<uint8_t*>(colorConvertedTensor.mutable_data_ptr()),
        nullptr,
        nullptr,
        nullptr};
    int srcLinesizes[4] = {config_.inputWidth * bytesPerPixel_, 0, 0, 0};

    uint8_t* dstPointers[4] = {
        static_cast<uint8_t*>(outputTensor.mutable_data_ptr()),
        nullptr,
        nullptr,
        nullptr};
    int expectedOutputWidth = static_cast<int>(outputTensor.sizes()[1]);
    int dstLinesizes[4] = {expectedOutputWidth * bytesPerPixel_, 0, 0, 0};

    colorConvertedHeight = sws_scale(
        resizeSwsContext_.get(),
        srcPointers,
        srcLinesizes,
        0,
        config_.inputHeight,
        dstPointers,
        dstLinesizes);
  }

  return colorConvertedHeight;
}

} // namespace facebook::torchcodec
