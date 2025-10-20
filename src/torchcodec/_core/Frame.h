// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/Metadata.h"
#include "src/torchcodec/_core/StreamOptions.h"

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
  torch::Tensor data;
  double ptsSeconds;
  double durationSeconds;
};

struct FrameBatchOutput {
  torch::Tensor data; // 4D: of shape NCHW or NHWC.
  torch::Tensor ptsSeconds; // 1D of shape (N,)
  torch::Tensor durationSeconds; // 1D of shape (N,)

  FrameBatchOutput(
      int64_t numFrames,
      const FrameDims& outputDims,
      const torch::Device& device);
};

struct AudioFramesOutput {
  torch::Tensor data; // shape is (numChannels, numSamples)
  double ptsSeconds;
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
torch::Tensor allocateEmptyHWCTensor(
    const FrameDims& frameDims,
    const torch::Device& device,
    std::optional<int> numFrames = std::nullopt);

} // namespace facebook::torchcodec
