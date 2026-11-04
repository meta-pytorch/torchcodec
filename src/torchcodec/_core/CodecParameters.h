// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>
#include <optional>
#include <vector>

#include "FFMPEGCommon.h"

namespace facebook::torchcodec {

// What a decoder is built from: a stream's AVCodecParameters, plus the
// properties of the stream and of its container that aren't in
// AVCodecParameters.
struct CodecParameters {
  explicit CodecParameters(
      const AVStream& av_stream,
      const AVInputFormat* container_format);

  UniqueAVCodecParameters codecpar;
  AVRational time_base = {};
  // libavformat's guess at the stream's frame rate.
  AVRational r_frame_rate = {};
  // Only set when codecpar doesn't carry the display matrix in its coded side
  // data, i.e. before FFmpeg 6.1, where it was on the AVStream instead. Read
  // it with get_display_matrix().
  std::optional<std::array<int32_t, 9>> display_matrix;
  // The format of the container the stream was demuxed from. Statically
  // allocated by FFmpeg.
  const AVInputFormat* container_format = nullptr;
};

// The display matrix, from codecpar's coded side data or from the
// `display_matrix` field, or nullptr when there is none. Points into
// `codec_parameters`.
const int32_t* get_display_matrix(const CodecParameters& codec_parameters);

// Degrees counter-clockwise needed to make the frames upright, from the display
// matrix.
std::optional<double> get_rotation(const CodecParameters& codec_parameters);

// The stream's side data, pointing into `codec_parameters`. That's
// AVCodecParameters.coded_side_data, plus the `display_matrix` field when it's
// set: before FFmpeg 6.1, the display matrix was on the AVStream, and it's the
// only one of the AVStream's side data that we keep.
std::vector<AVPacketSideData> get_stream_side_data(
    const CodecParameters& codec_parameters);

} // namespace facebook::torchcodec
