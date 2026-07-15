// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Metadata.h"
#include "StableABICompat.h"

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

std::optional<double> StreamMetadata::get_duration_seconds(
    SeekMode seek_mode) const {
  switch (seek_mode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      STD_TORCH_CHECK(
          end_stream_pts_seconds_from_content.has_value() &&
              begin_stream_pts_seconds_from_content.has_value(),
          "Missing beginStreamPtsSecondsFromContent or endStreamPtsSecondsFromContent");
      return end_stream_pts_seconds_from_content.value() -
          begin_stream_pts_seconds_from_content.value();
    case SeekMode::approximate:
      if (duration_seconds_from_header.has_value()) {
        return duration_seconds_from_header.value();
      }
      if (num_frames_from_header.has_value() &&
          average_fps_from_header.has_value() &&
          average_fps_from_header.value() != 0.0) {
        return static_cast<double>(num_frames_from_header.value()) /
            average_fps_from_header.value();
      }
      if (duration_seconds_from_container.has_value()) {
        return duration_seconds_from_container.value();
      }
      return std::nullopt;
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

double StreamMetadata::get_begin_stream_seconds(SeekMode seek_mode) const {
  switch (seek_mode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      STD_TORCH_CHECK(
          begin_stream_pts_seconds_from_content.has_value(),
          "Missing beginStreamPtsSecondsFromContent");
      return begin_stream_pts_seconds_from_content.value();
    case SeekMode::approximate:
      if (begin_stream_seconds_from_header.has_value()) {
        return begin_stream_seconds_from_header.value();
      }
      return 0.0;
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

std::optional<double> StreamMetadata::get_end_stream_seconds(
    SeekMode seek_mode) const {
  switch (seek_mode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      STD_TORCH_CHECK(
          end_stream_pts_seconds_from_content.has_value(),
          "Missing endStreamPtsSecondsFromContent");
      return end_stream_pts_seconds_from_content.value();
    case SeekMode::approximate: {
      auto dur = get_duration_seconds(seek_mode);
      if (dur.has_value()) {
        return get_begin_stream_seconds(seek_mode) + dur.value();
      }
      return std::nullopt;
    }
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

std::optional<int64_t> StreamMetadata::get_num_frames(
    SeekMode seek_mode) const {
  switch (seek_mode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      STD_TORCH_CHECK(
          num_frames_from_content.has_value(), "Missing numFramesFromContent");
      return num_frames_from_content.value();
    case SeekMode::approximate: {
      auto duration_seconds = get_duration_seconds(seek_mode);
      if (num_frames_from_header.has_value()) {
        return num_frames_from_header.value();
      }
      if (average_fps_from_header.has_value() && duration_seconds.has_value()) {
        return static_cast<int64_t>(
            average_fps_from_header.value() * duration_seconds.value());
      }
      return std::nullopt;
    }
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

std::optional<double> StreamMetadata::get_average_fps(
    SeekMode seek_mode) const {
  switch (seek_mode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact: {
      auto num_frames = get_num_frames(seek_mode);
      if (num_frames.has_value() &&
          begin_stream_pts_seconds_from_content.has_value() &&
          end_stream_pts_seconds_from_content.has_value()) {
        double duration = end_stream_pts_seconds_from_content.value() -
            begin_stream_pts_seconds_from_content.value();
        if (duration != 0.0) {
          return static_cast<double>(num_frames.value()) / duration;
        }
      }
      return average_fps_from_header;
    }
    case SeekMode::approximate:
      return average_fps_from_header;
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

std::optional<std::string> StreamMetadata::get_color_primaries_name() const {
  if (!color_primaries.has_value()) {
    return std::nullopt;
  }
  const char* name = av_color_primaries_name(*color_primaries);
  if (name == nullptr) {
    return std::nullopt;
  }
  return std::string(name);
}

std::optional<std::string> StreamMetadata::get_color_space_name() const {
  if (!color_space.has_value()) {
    return std::nullopt;
  }
  const char* name = av_color_space_name(*color_space);
  if (name == nullptr) {
    return std::nullopt;
  }
  return std::string(name);
}

std::optional<std::string>
StreamMetadata::get_color_transfer_characteristic_name() const {
  if (!color_transfer_characteristic.has_value()) {
    return std::nullopt;
  }
  const char* name = av_color_transfer_name(*color_transfer_characteristic);
  if (name == nullptr) {
    return std::nullopt;
  }
  return std::string(name);
}

} // namespace facebook::torchcodec
