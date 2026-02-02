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

// SwScale uses a double swscale path:
// 1. Color conversion (e.g., YUV -> RGB24) at the original frame resolution
// 2. Resize in RGB24 space (if resizing is needed)
//
// This approach ensures that transforms happen in the output color space
// (RGB24) rather than the input color space (YUV).
class SwScale {
 public:
  explicit SwScale(int swsFlags = SWS_BILINEAR);

  int convert(
      const UniqueAVFrame& avFrame,
      torch::Tensor& outputTensor,
      const FrameDims& outputDims);

 private:
  int swsFlags_;

  // Color conversion context (YUV -> RGB24). We cache this to avoid
  // recreating it for every frame.
  UniqueSwsContext colorConversionSwsContext_;
  SwsFrameContext prevColorConversionFrameContext_;

  // Resize context (RGB24 -> RGB24 at different resolution). We cache this
  // to avoid recreating it for every frame.
  UniqueSwsContext resizeSwsContext_;
  SwsFrameContext prevResizeFrameContext_;
};

} // namespace facebook::torchcodec
