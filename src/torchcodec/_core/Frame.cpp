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
    int64_t num_frames,
    const FrameDims& output_dims,
    const StableDevice& device,
    OutputDtype output_dtype)
    : pts_seconds(torch::stable::empty({num_frames}, kStableFloat64)),
      duration_seconds(torch::stable::empty({num_frames}, kStableFloat64)) {
  data =
      allocate_empty_hwc_tensor(output_dims, device, output_dtype, num_frames);
}

torch::stable::Tensor allocate_empty_hwc_tensor(
    const FrameDims& frame_dims,
    const StableDevice& device,
    OutputDtype output_dtype,
    std::optional<int> num_frames) {
  STD_TORCH_CHECK(
      frame_dims.height > 0, "height must be > 0, got: ", frame_dims.height);
  STD_TORCH_CHECK(
      frame_dims.width > 0, "width must be > 0, got: ", frame_dims.width);
  auto dtype =
      output_dtype == OutputDtype::FLOAT32 ? kStableUInt16 : kStableUInt8;
  if (num_frames.has_value()) {
    auto num_frames_value = num_frames.value();
    STD_TORCH_CHECK(
        num_frames_value >= 0,
        "numFrames must be >= 0, got: ",
        num_frames_value);
    return torch::stable::empty(
        {num_frames_value, frame_dims.height, frame_dims.width, 3},
        dtype,
        std::nullopt,
        device);
  } else {
    return torch::stable::empty(
        {frame_dims.height, frame_dims.width, 3}, dtype, std::nullopt, device);
  }
}

} // namespace facebook::torchcodec
