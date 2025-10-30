// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/SwsContext.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"

extern "C" {
#include <libswscale/swscale.h>
}

namespace facebook::torchcodec {

SwsFrameContext::SwsFrameContext(
    int inputWidth,
    int inputHeight,
    AVPixelFormat inputFormat,
    int outputWidth,
    int outputHeight)
    : inputWidth(inputWidth),
      inputHeight(inputHeight),
      inputFormat(inputFormat),
      outputWidth(outputWidth),
      outputHeight(outputHeight) {}

bool SwsFrameContext::operator==(const SwsFrameContext& other) const {
  return inputWidth == other.inputWidth && inputHeight == other.inputHeight &&
      inputFormat == other.inputFormat && outputWidth == other.outputWidth &&
      outputHeight == other.outputHeight;
}

bool SwsFrameContext::operator!=(const SwsFrameContext& other) const {
  return !(*this == other);
}

SwsContext* SwsScaler::getOrCreateContext(
    const UniqueAVFrame& avFrame,
    const FrameDims& outputDims,
    AVColorSpace colorspace,
    AVPixelFormat outputFormat,
    int swsFlags) {
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);

  SwsFrameContext currentFrameContext(
      avFrame->width,
      avFrame->height,
      frameFormat,
      outputDims.width,
      outputDims.height);

  // Recreate swscale context only if frame properties changed
  if (!swsContext_ || prevFrameContext_ != currentFrameContext) {
    swsContext_ = createSwsContext(
        currentFrameContext, colorspace, outputFormat, swsFlags);
    prevFrameContext_ = currentFrameContext;
  }

  return swsContext_.get();
}

} // namespace facebook::torchcodec
