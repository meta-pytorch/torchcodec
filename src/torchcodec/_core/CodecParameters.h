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

  // Degrees counter-clockwise needed to make the frames upright, from the
  // display matrix.
  std::optional<double> get_rotation() const;

  // The stream's side data, pointing into this object. From FFmpeg 6.1, that's
  // codecpar->coded_side_data. Before, it's only the display matrix, which was
  // on the AVStream: it's the only one of the AVStream's side data that we
  // keep.
  std::vector<AVPacketSideData> get_stream_side_data() const;

  UniqueAVCodecParameters codecpar;
  AVRational time_base = {};
  // libavformat's guess at the stream's frame rate.
  AVRational r_frame_rate = {};
  // The single source of truth for the display matrix: read it from here, not
  // from codecpar. It's filled from codecpar->coded_side_data from FFmpeg 6.1,
  // and from the AVStream's side data before, when AVCodecParameters had no
  // side data.
  std::optional<std::array<int32_t, 9>> display_matrix;
  // The format of the container the stream was demuxed from. Statically
  // allocated by FFmpeg.
  const AVInputFormat* container_format = nullptr;
};

} // namespace facebook::torchcodec
