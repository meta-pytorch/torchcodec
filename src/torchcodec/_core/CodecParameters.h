// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>

#include "FFMPEGCommon.h"

namespace facebook::torchcodec {

// A stream's AVCodecParameters, plus the properties of the stream that aren't
// in AVCodecParameters: its time base, r_frame_rate and rotation.
struct CodecParameters {
  UniqueAVCodecParameters codecpar;
  AVRational time_base = {};
  // libavformat's guess at the stream's frame rate.
  AVRational r_frame_rate = {};
  // From the stream's display matrix.
  std::optional<double> rotation;
};

} // namespace facebook::torchcodec
