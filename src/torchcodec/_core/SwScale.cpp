// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "SwScale.h"
#include "Frame.h"

namespace facebook::torchcodec {

SwScale::SwScale(int swsFlags) : swsFlags_(swsFlags) {}

int SwScale::convert(
    const UniqueAVFrame& avFrame,
    torch::Tensor& outputTensor,
    const FrameDims& outputDims) {
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);

  bool needsResize =
      (avFrame->height != outputDims.height ||
       avFrame->width != outputDims.width);

  // We need to compare the current frame context with our previous frame
  // context. If they are different, then we need to re-create our colorspace
  // conversion objects. We create our colorspace conversion objects late so
  // that we don't have to depend on the unreliable metadata in the header.
  // And we sometimes re-create them because it's possible for frame
  // resolution to change mid-stream. Finally, we want to reuse the colorspace
  // conversion objects as much as possible for performance reasons.
  SwsFrameContext colorConversionFrameContext(
      avFrame->width,
      avFrame->height,
      frameFormat,
      needsResize ? avFrame->width : outputDims.width,
      needsResize ? avFrame->height : outputDims.height);

  if (!colorConversionSwsContext_ ||
      prevColorConversionFrameContext_ != colorConversionFrameContext) {
    colorConversionSwsContext_ = createSwsContext(
        colorConversionFrameContext,
        avFrame->colorspace,

        // See [Transform and Format Conversion Order] for more on the output
        // pixel format.
        /*outputFormat=*/AV_PIX_FMT_RGB24,

        // No flags for color conversion. When resizing is needed, we use a
        // separate swscale context with the appropriate resize flags.
        /*swsFlags=*/0);
    prevColorConversionFrameContext_ = colorConversionFrameContext;
  }

  // When no resize is needed, we do color conversion directly into the output
  // tensor.

  torch::Tensor colorConvertedTensor = needsResize
      ? allocateEmptyHWCTensor(
            FrameDims(avFrame->height, avFrame->width), torch::kCPU)
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
    // Use cached swscale context for resizing, similar to the color conversion
    // context caching above.
    SwsFrameContext resizeFrameContext(
        avFrame->width,
        avFrame->height,
        AV_PIX_FMT_RGB24,
        outputDims.width,
        outputDims.height);

    if (!resizeSwsContext_ || prevResizeFrameContext_ != resizeFrameContext) {
      resizeSwsContext_ = createSwsContext(
          resizeFrameContext,
          AVCOL_SPC_RGB,
          /*outputFormat=*/AV_PIX_FMT_RGB24,
          /*swsFlags=*/swsFlags_);
      prevResizeFrameContext_ = resizeFrameContext;
    }

    uint8_t* srcPointers[4] = {
        colorConvertedTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
    int srcLinesizes[4] = {avFrame->width * 3, 0, 0, 0};

    uint8_t* dstPointers[4] = {
        outputTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
    int expectedOutputWidth = static_cast<int>(outputTensor.sizes()[1]);
    int dstLinesizes[4] = {expectedOutputWidth * 3, 0, 0, 0};

    colorConvertedHeight = sws_scale(
        resizeSwsContext_.get(),
        srcPointers,
        srcLinesizes,
        0,
        avFrame->height,
        dstPointers,
        dstLinesizes);
  }

  return colorConvertedHeight;
}

} // namespace facebook::torchcodec
