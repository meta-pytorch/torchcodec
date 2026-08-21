// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "AudioCommon.h"

#include <vector>

namespace facebook::torchcodec {

torch::headeronly::ScalarType sample_format_dtype(
    AVSampleFormat sample_format) {
  switch (av_get_packed_sample_fmt(sample_format)) {
    case AV_SAMPLE_FMT_U8:
      return kStableUInt8;
    case AV_SAMPLE_FMT_S16:
      return kStableInt16;
    case AV_SAMPLE_FMT_S32:
      return kStableInt32;
    case AV_SAMPLE_FMT_S64:
      return kStableInt64;
    case AV_SAMPLE_FMT_FLT:
      return kStableFloat32;
    case AV_SAMPLE_FMT_DBL:
      return kStableFloat64;
    default:
      break;
  }
  const char* name = av_get_sample_fmt_name(sample_format);
  STD_TORCH_CHECK(
      false,
      "Unsupported sample format '",
      name == nullptr ? "unknown" : name,
      "'.");
  return kStableUInt8;
}

AVSampleFormat planar_sample_format(torch::headeronly::ScalarType dtype) {
  switch (dtype) {
    case kStableUInt8:
      return AV_SAMPLE_FMT_U8P;
    case kStableInt16:
      return AV_SAMPLE_FMT_S16P;
    case kStableInt32:
      return AV_SAMPLE_FMT_S32P;
    case kStableInt64:
      return AV_SAMPLE_FMT_S64P;
    case kStableFloat32:
      return AV_SAMPLE_FMT_FLTP;
    case kStableFloat64:
      return AV_SAMPLE_FMT_DBLP;
    default:
      break;
  }
  STD_TORCH_CHECK(
      false,
      "Unsupported dtype for audio samples. Expected one of uint8, int16, "
      "int32, int64, float32 or float64.");
  return AV_SAMPLE_FMT_NONE;
}

torch::stable::Tensor swr_convert_to_tensor(
    const UniqueSwrContext& swr_context,
    const uint8_t* const* src_planes,
    int num_src_samples,
    int num_out_channels,
    int64_t num_out_samples_bound) {
  torch::stable::Tensor out = torch::stable::empty(
      {num_out_channels, num_out_samples_bound}, kStableFloat32);
  if (num_out_samples_bound == 0) {
    return out;
  }

  // A contiguous [num_out_channels, N] float32 tensor is exactly FLTP: one
  // plane per row.
  int64_t bytes_per_channel =
      num_out_samples_bound * av_get_bytes_per_sample(kAudioOutSampleFormat);
  auto* base = static_cast<uint8_t*>(out.mutable_data_ptr());
  std::vector<uint8_t*> out_planes(num_out_channels);
  for (int channel = 0; channel < num_out_channels; ++channel) {
    out_planes[channel] = base + channel * bytes_per_channel;
  }

  int num_out_samples = swr_convert(
      swr_context.get(),
      out_planes.data(),
      static_cast<int>(num_out_samples_bound),
      src_planes,
      num_src_samples);
  STD_TORCH_CHECK(
      num_out_samples >= 0,
      "Error in swr_convert: ",
      get_ffmpeg_error_string_from_error_code(num_out_samples));

  return torch::stable::narrow(
      out, /*dim=*/1, /*start=*/0, /*length=*/num_out_samples);
}

} // namespace facebook::torchcodec
