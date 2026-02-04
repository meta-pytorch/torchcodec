// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include <optional>
#include <string>

namespace facebook::torchcodec {

struct WavSamples {
  torch::Tensor samples; // Shape: (num_channels, num_samples)
  std::string metadataJson; // JSON compatible with AudioStreamMetadata
};

// Validate parameters and decode WAV from bytes tensor.
// Returns nullopt if:
//   - The data is not a valid/supported WAV file
//   - stream_index is specified and != 0 (WAV only has one stream)
//   - sample_rate is specified and doesn't match the file's sample rate
//   - num_channels is specified and doesn't match the file's channel count
std::optional<WavSamples> validateAndDecodeWavFromTensor(
    const torch::Tensor& data,
    std::optional<int64_t> stream_index = std::nullopt,
    std::optional<int64_t> sample_rate = std::nullopt,
    std::optional<int64_t> num_channels = std::nullopt);

// Validate parameters and decode WAV from file path.
// Returns nullopt if:
//   - The file is not a valid/supported WAV file
//   - stream_index is specified and != 0 (WAV only has one stream)
//   - sample_rate is specified and doesn't match the file's sample rate
//   - num_channels is specified and doesn't match the file's channel count
std::optional<WavSamples> validateAndDecodeWavFromFile(
    const std::string& path,
    std::optional<int64_t> stream_index = std::nullopt,
    std::optional<int64_t> sample_rate = std::nullopt,
    std::optional<int64_t> num_channels = std::nullopt);

} // namespace facebook::torchcodec
