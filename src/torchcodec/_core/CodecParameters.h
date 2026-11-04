// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>
#include <optional>

#include "FFMPEGCommon.h"

namespace facebook::torchcodec {

// What a decoder is built from: a stream's AVCodecParameters, plus the
// properties of the stream and of its container that aren't in
// AVCodecParameters.
struct CodecParameters {
  // A copy of `av_stream`'s codec parameters. `container_format` is the format
  // of the container `av_stream` belongs to.
  explicit CodecParameters(
      const AVStream& av_stream,
      const AVInputFormat* container_format);

  UniqueAVCodecParameters codecpar;
  AVRational time_base = {};
  // libavformat's guess at the stream's frame rate.
  AVRational r_frame_rate = {};
  std::optional<std::array<int32_t, 9>> display_matrix;
  // The format of the container the stream was demuxed from. Statically
  // allocated by FFmpeg.
  const AVInputFormat* container_format = nullptr;
};

// Degrees counter-clockwise needed to make the frames upright, from the display
// matrix.
std::optional<double> get_rotation(const CodecParameters& codec_parameters);

} // namespace facebook::torchcodec
