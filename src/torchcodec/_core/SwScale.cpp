// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "SwScale.h"
#include "Frame.h"

namespace facebook::torchcodec {

SwScaleContext::SwScaleContext(
    int inputWidth,
    int inputHeight,
    AVPixelFormat inputFormat,
    AVColorSpace inputColorspace,
    int outputWidth,
    int outputHeight)
    : inputWidth(inputWidth),
      inputHeight(inputHeight),
      inputFormat(inputFormat),
      inputColorspace(inputColorspace),
      outputWidth(outputWidth),
      outputHeight(outputHeight) {}

bool SwScaleContext::operator==(const SwScaleContext& other) const {
  return inputWidth == other.inputWidth && inputHeight == other.inputHeight &&
      inputFormat == other.inputFormat &&
      inputColorspace == other.inputColorspace &&
      outputWidth == other.outputWidth && outputHeight == other.outputHeight;
}

bool SwScaleContext::operator!=(const SwScaleContext& other) const {
  return !(*this == other);
}

SwScale::SwScale(const SwScaleContext& context, int swsFlags)
    : context_(context), swsFlags_(swsFlags) {
  bool needsResize =
      (context_.inputHeight != context_.outputHeight ||
       context_.inputWidth != context_.outputWidth);

  // Create color conversion context (input format -> RGB24).
  // When resizing is needed, color conversion outputs at input resolution.
  // When no resize is needed, color conversion outputs at output resolution.
  SwsFrameContext colorConversionFrameContext(
      context_.inputWidth,
      context_.inputHeight,
      context_.inputFormat,
      needsResize ? context_.inputWidth : context_.outputWidth,
      needsResize ? context_.inputHeight : context_.outputHeight);

  colorConversionSwsContext_ = createSwsContext(
      colorConversionFrameContext,
      context_.inputColorspace,
      // See [Transform and Format Conversion Order] for more on the output
      // pixel format.
      /*outputFormat=*/AV_PIX_FMT_RGB24,
      // No flags for color conversion. When resizing is needed, we use a
      // separate swscale context with the appropriate resize flags.
      /*swsFlags=*/0);

  // Create resize context if needed (RGB24 at input resolution -> RGB24 at
  // output resolution).
  if (needsResize) {
    SwsFrameContext resizeFrameContext(
        context_.inputWidth,
        context_.inputHeight,
        AV_PIX_FMT_RGB24,
        context_.outputWidth,
        context_.outputHeight);

    resizeSwsContext_ = createSwsContext(
        resizeFrameContext,
        AVCOL_SPC_RGB,
        /*outputFormat=*/AV_PIX_FMT_RGB24,
        /*swsFlags=*/swsFlags_);
  }
}

int SwScale::convert(
    const UniqueAVFrame& avFrame,
    torch::Tensor& outputTensor) {
  bool needsResize =
      (context_.inputHeight != context_.outputHeight ||
       context_.inputWidth != context_.outputWidth);

  // When no resize is needed, we do color conversion directly into the output
  // tensor. When resize is needed, we first convert to an intermediate tensor
  // at the input resolution, then resize into the output tensor.
  torch::Tensor colorConvertedTensor = needsResize
      ? allocateEmptyHWCTensor(
            FrameDims(context_.inputHeight, context_.inputWidth), torch::kCPU)
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

  if (needsResize) {
    uint8_t* srcPointers[4] = {
        colorConvertedTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
    int srcLinesizes[4] = {context_.inputWidth * 3, 0, 0, 0};

    uint8_t* dstPointers[4] = {
        outputTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
    int expectedOutputWidth = static_cast<int>(outputTensor.sizes()[1]);
    int dstLinesizes[4] = {expectedOutputWidth * 3, 0, 0, 0};

    colorConvertedHeight = sws_scale(
        resizeSwsContext_.get(),
        srcPointers,
        srcLinesizes,
        0,
        context_.inputHeight,
        dstPointers,
        dstLinesizes);
  }

  return colorConvertedHeight;
}

} // namespace facebook::torchcodec
