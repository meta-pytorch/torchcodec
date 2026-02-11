// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Frame.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

FrameDims::FrameDims(int height, int width) : height(height), width(width) {
  STD_TORCH_CHECK(height > 0, "FrameDims.height must be > 0, got: ", height);
  STD_TORCH_CHECK(width > 0, "FrameDims.width must be > 0, got: ", width);
}

FrameBatchOutput::FrameBatchOutput(
    int64_t numFrames,
    const FrameDims& outputDims,
    const StableDevice& device)
    : ptsSeconds(stableEmptyCPU({numFrames}, kStableFloat64)),
      durationSeconds(stableEmptyCPU({numFrames}, kStableFloat64)) {
  data = allocateEmptyHWCTensor(outputDims, device, numFrames);
}

StableTensor allocateEmptyHWCTensor(
    const FrameDims& frameDims,
    const StableDevice& device,
    std::optional<int> numFrames) {
  STD_TORCH_CHECK(
      frameDims.height > 0, "height must be > 0, got: ", frameDims.height);
  STD_TORCH_CHECK(
      frameDims.width > 0, "width must be > 0, got: ", frameDims.width);
  if (numFrames.has_value()) {
    auto numFramesValue = numFrames.value();
    STD_TORCH_CHECK(
        numFramesValue >= 0, "numFrames must be >= 0, got: ", numFramesValue);
    return stableEmpty(
        {numFramesValue, frameDims.height, frameDims.width, 3},
        kStableUInt8,
        device);
  } else {
    return stableEmpty(
        {frameDims.height, frameDims.width, 3}, kStableUInt8, device);
  }
}

} // namespace facebook::torchcodec
