// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>

#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

// Audio conversion building block: turns a decoded frame's samples, in the
// codec's own sample type, into normalized float32 ones - optionally at another
// sample rate, and with another channel count.

// Not thread-safe.
class FORCE_PUBLIC_VISIBILITY AudioConverter {
 public:
  explicit AudioConverter(
      std::optional<int> sample_rate = std::nullopt,
      std::optional<int> num_channels = std::nullopt);

  torch::stable::Tensor convert(
      const torch::stable::Tensor& samples,
      int sample_rate);

  torch::stable::Tensor drain();

  void reset();

  bool has_converted_samples() const {
    return swr_context_ != nullptr;
  }

  // Only meaningful once convert() has been called at least once.
  int out_sample_rate() const {
    return out_sample_rate_;
  }

 private:
  std::optional<int> requested_sample_rate_;
  std::optional<int> requested_num_channels_;

  UniqueSwrContext swr_context_;
  // All of these are set when swr_context_ is created, and describe what it was
  // configured for. The src_ ones are what a later convert() is checked
  // against: swresample is built once for one input shape.
  AVSampleFormat src_sample_format_ = AV_SAMPLE_FMT_NONE;
  int src_sample_rate_ = 0;
  int src_num_channels_ = 0;
  int out_sample_rate_ = 0;
  int out_num_channels_ = 0;
};

} // namespace facebook::torchcodec
