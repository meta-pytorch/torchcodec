// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "AudioConverter.h"

#include <vector>

#include "AudioCommon.h"

namespace facebook::torchcodec {

AudioConverter::AudioConverter(
    std::optional<int> sample_rate,
    std::optional<int> num_channels)
    : requested_sample_rate_(sample_rate),
      requested_num_channels_(num_channels) {
  STD_TORCH_CHECK(
      !sample_rate.has_value() || *sample_rate > 0,
      "sample_rate must be > 0. Got: ",
      sample_rate.value_or(0));
  STD_TORCH_CHECK(
      !num_channels.has_value() || *num_channels > 0,
      "num_channels must be > 0. Got: ",
      num_channels.value_or(0));
}

void AudioConverter::reset() {
  swr_context_.reset();
  src_sample_format_ = AV_SAMPLE_FMT_NONE;
  src_sample_rate_ = 0;
  src_num_channels_ = 0;
  out_sample_rate_ = 0;
  out_num_channels_ = 0;
}

torch::stable::Tensor AudioConverter::convert(
    const torch::stable::Tensor& samples,
    int sample_rate) {
  STD_TORCH_CHECK(
      samples.dim() == 2,
      "Expected a 2D [num_channels, num_samples] tensor, got a ",
      samples.dim(),
      "D one.");
  STD_TORCH_CHECK(samples.is_contiguous(), "The samples must be contiguous.");
  STD_TORCH_CHECK(sample_rate > 0, "sample_rate must be > 0.");

  AVSampleFormat src_sample_format =
      planar_sample_format(samples.scalar_type());
  int num_channels = static_cast<int>(samples.sizes()[0]);
  int num_samples = static_cast<int>(samples.sizes()[1]);
  STD_TORCH_CHECK(
      num_channels > 0, "The samples must have at least 1 channel.");

  if (swr_context_ == nullptr) {
    src_sample_format_ = src_sample_format;
    src_sample_rate_ = sample_rate;
    src_num_channels_ = num_channels;
    out_sample_rate_ = requested_sample_rate_.value_or(sample_rate);
    out_num_channels_ = requested_num_channels_.value_or(num_channels);
    swr_context_.reset(create_swr_context(
        src_sample_format_,
        kAudioOutSampleFormat,
        src_sample_rate_,
        out_sample_rate_,
        src_num_channels_,
        out_num_channels_));
  } else {
    // swresample is configured once, from the first samples we see, and its
    // buffered state is tied to that configuration. Rather than silently
    // reconfiguring - which would discard whatever it still holds - we make the
    // caller decide, by reset()ing.
    STD_TORCH_CHECK(
        src_sample_format == src_sample_format_ &&
            sample_rate == src_sample_rate_ &&
            num_channels == src_num_channels_,
        "This AudioConverter was set up for ",
        src_num_channels_,
        " channels of ",
        av_get_sample_fmt_name(src_sample_format_),
        " at ",
        src_sample_rate_,
        " Hz, but got ",
        num_channels,
        " channels of ",
        av_get_sample_fmt_name(src_sample_format),
        " at ",
        sample_rate,
        " Hz. Call reset() to convert a different stream.");
  }

  const auto* base = static_cast<const uint8_t*>(samples.const_data_ptr());
  int64_t bytes_per_channel =
      num_samples * av_get_bytes_per_sample(src_sample_format);
  std::vector<const uint8_t*> src_planes(num_channels);
  for (int channel = 0; channel < num_channels; ++channel) {
    src_planes[channel] = base + channel * bytes_per_channel;
  }

  return swr_convert_to_tensor(
      swr_context_,
      src_planes.data(),
      num_samples,
      out_num_channels_,
      get_swr_output_num_samples_bound(
          swr_context_, num_samples, src_sample_rate_, out_sample_rate_));
}

torch::stable::Tensor AudioConverter::drain() {
  STD_TORCH_CHECK(
      swr_context_ != nullptr,
      "This AudioConverter hasn't converted any samples, so there is nothing "
      "to drain and no way to know what shape the result should have.");
  // A null input is what tells swr_convert() to flush. Unlike the convert()
  // path we ask swresample how much it is holding rather than deriving a bound
  // from an input size, since here there is no input.
  return swr_convert_to_tensor(
      swr_context_,
      /*src_planes=*/nullptr,
      /*num_src_samples=*/0,
      out_num_channels_,
      swr_get_out_samples(swr_context_.get(), 0));
}

} // namespace facebook::torchcodec
