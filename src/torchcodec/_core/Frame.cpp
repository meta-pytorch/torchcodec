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
    : ptsSeconds(torch::empty({numFrames}, {torch::kFloat64})),
      durationSeconds(torch::empty({numFrames}, {torch::kFloat64})) {
  data = allocateEmptyHWCTensor(outputDims, device, numFrames);
}

torch::Tensor allocateEmptyHWCTensor(
    const FrameDims& frameDims,
    const StableDevice& device,
    std::optional<int> numFrames) {
  torch::Device torchDevice(
      static_cast<c10::DeviceType>(device.type()), device.index());
  auto tensorOptions = torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .layout(torch::kStrided)
                           .device(torchDevice);
  STD_TORCH_CHECK(
      frameDims.height > 0, "height must be > 0, got: ", frameDims.height);
  STD_TORCH_CHECK(
      frameDims.width > 0, "width must be > 0, got: ", frameDims.width);
  if (numFrames.has_value()) {
    auto numFramesValue = numFrames.value();
    STD_TORCH_CHECK(
        numFramesValue >= 0, "numFrames must be >= 0, got: ", numFramesValue);
    return torch::empty(
        {numFramesValue, frameDims.height, frameDims.width, 3}, tensorOptions);
  } else {
    return torch::empty({frameDims.height, frameDims.width, 3}, tensorOptions);
  }
}

} // namespace facebook::torchcodec
