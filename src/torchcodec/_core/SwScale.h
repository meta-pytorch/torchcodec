// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

struct FrameDims;

// SwScale uses a double swscale path:
// 1. Color conversion (e.g., YUV -> RGB24/RGB48) at the original frame
//    resolution
// 2. Resize in output RGB space (if resizing is needed)
//
// This approach ensures that transforms happen in the output color space
// (RGB) rather than the input color space (YUV).
//
// The caller is responsible for caching SwScale instances and recreating them
// when the context changes, similar to how FilterGraph is managed.
class SwScale {
 public:
  // config.outputFormat is AV_PIX_FMT_RGB24 for 8-bit, AV_PIX_FMT_RGB48 for
  // >8-bit.
  SwScale(const SwsConfig& config, int swsFlags = SWS_BILINEAR);

  int convert(
      const UniqueAVFrame& avFrame,
      torch::stable::Tensor& outputTensor);

  const SwsConfig& getConfig() const {
    return config_;
  }

 private:
  SwsConfig config_;
  int swsFlags_;
  bool needsResize_;

  // Bytes per pixel for the output format (3 for RGB24, 6 for RGB48).
  int bytesPerPixel_;

  // Color conversion context (input format -> output RGB at original
  // resolution).
  UniqueSwsContext colorConversionSwsContext_;

  // Resize context (output RGB at input res -> output RGB at output res).
  // May be null if no resize is needed.
  UniqueSwsContext resizeSwsContext_;
};

} // namespace facebook::torchcodec
