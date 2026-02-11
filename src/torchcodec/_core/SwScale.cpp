// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "SwScale.h"
#include "Frame.h"

namespace facebook::torchcodec {

SwScale::SwScale(const SwsFrameConfig& config, int swsFlags)
    : config_(config), swsFlags_(swsFlags) {
  needsResize_ =
      (config_.inputHeight != config_.outputHeight ||
       config_.inputWidth != config_.outputWidth);

  // Create color conversion context (input format -> RGB24).
  // Color conversion always outputs at the input resolution.
  // When no resize is needed, input and output resolutions are the same.
  SwsFrameConfig colorConversionFrameConfig(
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
      /*outputFormat=*/AV_PIX_FMT_RGB24,
      // No flags for color conversion. When resizing is needed, we use a
      // separate swscale context with the appropriate resize flags.
      /*swsFlags=*/0);

  // Create resize context if needed (RGB24 at input resolution -> RGB24 at
  // output resolution).
  if (needsResize_) {
    SwsFrameConfig resizeFrameConfig(
        config_.inputWidth,
        config_.inputHeight,
        AV_PIX_FMT_RGB24,
        AVCOL_SPC_RGB,
        config_.outputWidth,
        config_.outputHeight);

    resizeSwsContext_ = createSwsContext(
        resizeFrameConfig,
        /*outputFormat=*/AV_PIX_FMT_RGB24,
        /*swsFlags=*/swsFlags_);
  }
}

int SwScale::convert(
    const UniqueAVFrame& avFrame,
    torch::Tensor& outputTensor) {
  // When resizing is needed, we do sws_scale twice: first convert to RGB24 at
  // original resolution, then resize in RGB24 space. This ensures transforms
  // happen in the output color space (RGB24) rather than the input color space
  // (YUV).
  //
  // When no resize is needed, we do color conversion directly into the output
  // tensor.
  torch::Tensor colorConvertedTensor = needsResize_
      ? allocateEmptyHWCTensor(
            FrameDims(config_.inputHeight, config_.inputWidth), torch::kCPU)
      : outputTensor;

  uint8_t* colorConvertedPointers[4] = {
      colorConvertedTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
  int colorConvertedWidth = static_cast<int>(colorConvertedTensor.sizes()[1]);
  int colorConvertedLinesizes[4] = {colorConvertedWidth * 3, 0, 0, 0};

  int colorConvertedHeight = sws_scale(
      colorConversionSwsContext_.get(),
      avFrame->data,
      avFrame->linesize,
      0,
      avFrame->height,
      colorConvertedPointers,
      colorConvertedLinesizes);

  TORCH_CHECK(
      colorConvertedHeight == avFrame->height,
      "Color conversion swscale pass failed: colorConvertedHeight != avFrame->height: ",
      colorConvertedHeight,
      " != ",
      avFrame->height);

  if (needsResize_) {
    uint8_t* srcPointers[4] = {
        colorConvertedTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
    int srcLinesizes[4] = {config_.inputWidth * 3, 0, 0, 0};

    uint8_t* dstPointers[4] = {
        outputTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
    int expectedOutputWidth = static_cast<int>(outputTensor.sizes()[1]);
    int dstLinesizes[4] = {expectedOutputWidth * 3, 0, 0, 0};

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
