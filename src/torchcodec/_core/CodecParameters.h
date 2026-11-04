// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

// A stream's AVCodecParameters, plus its time base, r_frame_rate and rotation:
// the fields of the Python VideoCodecParameters and AudioCodecParameters, with
// the same names. A field that is at its FFmpeg default is unset. Enums are
// FFmpeg's names for them.
struct CodecParameters {
  std::string codec;
  AVRational time_base = {};
  std::optional<torch::stable::Tensor> extradata; // uint8 [N]
  std::optional<int64_t> codec_tag;
  std::optional<int64_t> profile;
  std::optional<int64_t> level;
  std::optional<int64_t> bit_rate;
  std::optional<int64_t> bits_per_coded_sample;
  std::optional<int64_t> bits_per_raw_sample;
  // Parallel vectors, one entry per side data type.
  std::vector<std::string> coded_side_data_names;
  std::vector<torch::stable::Tensor> coded_side_data_values; // uint8 [N] each

  // Video only.
  std::optional<int64_t> width;
  std::optional<int64_t> height;
  std::optional<std::string> pixel_format;
  std::optional<AVRational> sample_aspect_ratio;
  std::optional<AVRational> framerate;
  std::optional<AVRational> r_frame_rate;
  std::optional<std::string> field_order;
  std::optional<std::string> color_range;
  std::optional<std::string> color_primaries;
  std::optional<std::string> color_transfer_characteristic;
  std::optional<std::string> color_space;
  std::optional<std::string> chroma_location;
  std::optional<int64_t> video_delay;
  std::optional<double> rotation;

  // Audio only.
  std::optional<std::string> sample_format;
  std::optional<int64_t> sample_rate;
  std::optional<std::string> channel_layout;
  std::optional<int64_t> block_align;
  std::optional<int64_t> frame_size;
  std::optional<int64_t> initial_padding;
  std::optional<int64_t> trailing_padding;
  std::optional<int64_t> seek_preroll;
};

CodecParameters get_codec_parameters(const AVStream& av_stream);

} // namespace facebook::torchcodec
