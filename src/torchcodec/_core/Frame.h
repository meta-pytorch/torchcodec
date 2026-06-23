// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "FFMPEGCommon.h"
#include "Metadata.h"
#include "StableABICompat.h"
#include "StreamOptions.h"

namespace facebook::torchcodec {

struct FrameDims {
  int height = 0;
  int width = 0;

  FrameDims() = default;

  FrameDims(int h, int w);
};

// All public video decoding entry points return either a FrameOutput or a
// FrameBatchOutput.
// They are the equivalent of the user-facing Frame and FrameBatch classes in
// Python. They contain RGB decoded frames along with some associated data
// like PTS and duration.
// FrameOutput is also relevant for audio decoding, typically as the output of
// getNextFrame(), or as a temporary output variable.
struct FrameOutput {
  // data shape is:
  // - 3D (C, H, W) or (H, W, C) for videos
  // - 2D (numChannels, numSamples) for audio
  torch::stable::Tensor data;
  double pts_seconds;
  double duration_seconds;
};

struct FrameBatchOutput {
  torch::stable::Tensor data; // 4D: of shape NCHW or NHWC.
  torch::stable::Tensor pts_seconds; // 1D of shape (N,)
  torch::stable::Tensor duration_seconds; // 1D of shape (N,)

  FrameBatchOutput(
      int64_t num_frames,
      const FrameDims& output_dims,
      const StableDevice& device,
      OutputDtype output_dtype);
};

struct AudioFramesOutput {
  torch::stable::Tensor data; // shape is (numChannels, numSamples)
  double pts_seconds;
};

// --------------------------------------------------------------------------
// FRAME TENSOR ALLOCATION APIs
// --------------------------------------------------------------------------

// Note [Frame Tensor allocation]
//
// We always allocate [N]HWC tensors. The low-level decoding functions all
// assume HWC tensors, since this is what FFmpeg natively handles. It's up to
// the high-level decoding entry-points to permute that back to CHW, by calling
// maybePermuteHWC2CHW().
torch::stable::Tensor allocate_empty_hwc_tensor(
    const FrameDims& frame_dims,
    const StableDevice& device,
    OutputDtype output_dtype,
    std::optional<int> num_frames = std::nullopt);

} // namespace facebook::torchcodec
