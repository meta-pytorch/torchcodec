// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Frame.h"
#include "TCError.h"

namespace facebook::torchcodec {

FrameDims::FrameDims(int height, int width) : height(height), width(width) {
  TC_CHECK(height > 0, "FrameDims.height must be > 0, got: ", height);
  TC_CHECK(width > 0, "FrameDims.width must be > 0, got: ", width);
}

FrameBatchOutput::FrameBatchOutput(
    int64_t numFrames,
    const FrameDims& outputDims,
    const tc::Device& device,
    OutputDtype outputDtype)
    : ptsSeconds(tc::empty({numFrames}, tc::kFloat64)),
      durationSeconds(tc::empty({numFrames}, tc::kFloat64)) {
  data = allocateEmptyHWCTensor(outputDims, device, outputDtype, numFrames);
}

tc::Tensor allocateEmptyHWCTensor(
    const FrameDims& frameDims,
    const tc::Device& device,
    OutputDtype outputDtype,
    std::optional<int> numFrames) {
  TC_CHECK(
      frameDims.height > 0, "height must be > 0, got: ", frameDims.height);
  TC_CHECK(
      frameDims.width > 0, "width must be > 0, got: ", frameDims.width);
  auto dtype = outputDtype == OutputDtype::FLOAT32 ? tc::kUInt16 : tc::kUInt8;
  if (numFrames.has_value()) {
    auto numFramesValue = numFrames.value();
    TC_CHECK(
        numFramesValue >= 0, "numFrames must be >= 0, got: ", numFramesValue);
    return tc::empty(
        {numFramesValue, frameDims.height, frameDims.width, 3}, dtype, device);
  } else {
    return tc::empty({frameDims.height, frameDims.width, 3}, dtype, device);
  }
}

} // namespace facebook::torchcodec
