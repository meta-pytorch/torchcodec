// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include "FFMPEGCommon.h"

namespace facebook::torchcodec {

struct FrameDims;

struct SwScaleContext {
  int inputWidth = 0;
  int inputHeight = 0;
  AVPixelFormat inputFormat = AV_PIX_FMT_NONE;
  AVColorSpace inputColorspace = AVCOL_SPC_UNSPECIFIED;
  int outputWidth = 0;
  int outputHeight = 0;

  SwScaleContext() = default;
  SwScaleContext(
      int inputWidth,
      int inputHeight,
      AVPixelFormat inputFormat,
      AVColorSpace inputColorspace,
      int outputWidth,
      int outputHeight);

  bool operator==(const SwScaleContext&) const;
  bool operator!=(const SwScaleContext&) const;
};

// SwScale uses a double swscale path:
// 1. Color conversion (e.g., YUV -> RGB24) at the original frame resolution
// 2. Resize in RGB24 space (if resizing is needed)
//
// This approach ensures that transforms happen in the output color space
// (RGB24) rather than the input color space (YUV).
//
// The caller is responsible for caching SwScale instances and recreating them
// when the context changes, similar to how FilterGraph is managed.
class SwScale {
 public:
  SwScale(const SwScaleContext& context, int swsFlags = SWS_BILINEAR);

  int convert(const UniqueAVFrame& avFrame, torch::Tensor& outputTensor);

 private:
  SwScaleContext context_;
  int swsFlags_;

  // Color conversion context (input format -> RGB24 at original resolution).
  UniqueSwsContext colorConversionSwsContext_;

  // Resize context (RGB24 -> RGB24 at output resolution).
  // May be null if no resize is needed.
  UniqueSwsContext resizeSwsContext_;
};

} // namespace facebook::torchcodec
