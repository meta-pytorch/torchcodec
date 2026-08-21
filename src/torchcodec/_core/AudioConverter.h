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
//
// Unlike ColorConverter this is a stream processor, not a function of its
// input, because resampling is an interpolation filter: the sample it emits at
// a given instant is a weighted sum of input samples on both sides of it. So
// swresample holds the tail of each frame back until the next one arrives,
// which means convert() emits fewer samples than it was given (sometimes none),
// drain() is needed to get the last ones out, and frames must be fed in order.
// reset() drops that state, and is what a caller must do after seeking.
//
// Not thread-safe.
class FORCE_PUBLIC_VISIBILITY AudioConverter {
 public:
  // Both default to the source's own value, i.e. to no conversion. Note that
  // the sample *type* is always converted, to float32.
  explicit AudioConverter(
      std::optional<int> sample_rate = std::nullopt,
      std::optional<int> num_channels = std::nullopt);

  // `samples` is a contiguous [num_channels, num_samples] tensor in the
  // source's own sample type, i.e. exactly what audio_samples() produces.
  // Returns the samples that are now computable, as float32
  // [out_num_channels, N] - and N may well be 0.
  torch::stable::Tensor convert(
      const torch::stable::Tensor& samples,
      int sample_rate);

  // The samples swresample was still holding on to. Callers who skip this lose
  // the tail of the stream.
  torch::stable::Tensor drain();

  // Drop the resampler's buffered state and start over, so that the next
  // convert() call reconfigures from the samples it is given.
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
