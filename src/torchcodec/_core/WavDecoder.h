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

// Decode WAV from bytes tensor (zero-copy when possible for mono PCM).
// Returns nullopt if the data is not a valid/supported WAV file.
std::optional<WavSamples> decodeWavFromTensor(const torch::Tensor& data);

// Decode WAV from file path (uses mmap for zero-copy access).
// Returns nullopt if the file is not a valid/supported WAV file.
std::optional<WavSamples> decodeWavFromFile(const std::string& path);

} // namespace facebook::torchcodec
