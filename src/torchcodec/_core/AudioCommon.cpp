// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "AudioCommon.h"

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

} // namespace facebook::torchcodec
