// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/Metadata.h"

#include <gtest/gtest.h>

namespace facebook::torchcodec {

// Test that num_frames_from_content always has priority when accessing
// getNumFrames()
TEST(MetadataTest, NumFramesFallbackPriority) {
  // in exact mode, both header and content available
  {
    StreamMetadata metadata;
    metadata.num_frames_from_header = 10;
    metadata.num_frames_from_content = 20;
    metadata.duration_seconds_from_header = 4.0;
    metadata.average_fps_from_header = 30.0;

    EXPECT_EQ(metadata.get_num_frames(SeekMode::exact), 20);
  }

  // in exact mode, only content available
  {
    StreamMetadata metadata;
    metadata.num_frames_from_header = std::nullopt;
    metadata.num_frames_from_content = 10;
    metadata.duration_seconds_from_header = 4.0;
    metadata.average_fps_from_header = 30.0;

    EXPECT_EQ(metadata.get_num_frames(SeekMode::exact), 10);
  }

  // in approximate mode, header should be used
  {
    StreamMetadata metadata;
    metadata.num_frames_from_header = 10;
    metadata.num_frames_from_content = std::nullopt;
    metadata.duration_seconds_from_header = 4.0;
    metadata.average_fps_from_header = 30.0;

    EXPECT_EQ(metadata.get_num_frames(SeekMode::approximate), 10);
  }
}

// Test that if num_frames_from_content and num_frames_from_header are missing,
// getNumFrames() is calculated using average_fps_from_header and
// duration_seconds_from_header in approximate mode
TEST(MetadataTest, CalculateNumFramesUsingFpsAndDuration) {
  // both fps and duration available
  {
    StreamMetadata metadata;
    metadata.num_frames_from_header = std::nullopt;
    metadata.num_frames_from_content = std::nullopt;
    metadata.average_fps_from_header = 60.0;
    metadata.duration_seconds_from_header = 10.0;

    EXPECT_EQ(metadata.get_num_frames(SeekMode::approximate), 600);
  }

  // fps available but duration missing
  {
    StreamMetadata metadata;
    metadata.num_frames_from_header = std::nullopt;
    metadata.num_frames_from_content = std::nullopt;
    metadata.average_fps_from_header = 60.0;
    metadata.duration_seconds_from_header = std::nullopt;

    EXPECT_EQ(metadata.get_num_frames(SeekMode::approximate), std::nullopt);
  }

  // duration available but fps missing
  {
    StreamMetadata metadata;
    metadata.num_frames_from_header = std::nullopt;
    metadata.num_frames_from_content = std::nullopt;
    metadata.average_fps_from_header = std::nullopt;
    metadata.duration_seconds_from_header = 10.0;

    EXPECT_EQ(metadata.get_num_frames(SeekMode::approximate), std::nullopt);
  }

  // both missing
  {
    StreamMetadata metadata;
    metadata.num_frames_from_header = std::nullopt;
    metadata.num_frames_from_content = std::nullopt;
    metadata.average_fps_from_header = std::nullopt;
    metadata.duration_seconds_from_header = std::nullopt;

    EXPECT_EQ(metadata.get_num_frames(SeekMode::approximate), std::nullopt);
  }
}

// Test that using begin_stream_seconds_from_content and
// end_stream_seconds_from_content to calculate getDurationSeconds() has
// priority. If either value is missing, duration_seconds_from_header is used.
TEST(MetadataTest, DurationSecondsFallback) {
  // in exact mode, both begin and end content available, should calculate from
  // them
  {
    StreamMetadata metadata;
    metadata.duration_seconds_from_header = 60.0;
    metadata.begin_stream_pts_seconds_from_content = 5.0;
    metadata.end_stream_pts_seconds_from_content = 20.0;

    EXPECT_NEAR(
        metadata.get_duration_seconds(SeekMode::exact).value(), 15.0, 1e-6);
  }

  // in exact mode, only content values, no header
  {
    StreamMetadata metadata;
    metadata.duration_seconds_from_header = std::nullopt;
    metadata.begin_stream_pts_seconds_from_content = 0.0;
    metadata.end_stream_pts_seconds_from_content = 10.0;

    EXPECT_NEAR(
        metadata.get_duration_seconds(SeekMode::exact).value(), 10.0, 1e-6);
  }

  // in approximate mode, header value takes priority (ignores content)
  {
    StreamMetadata metadata;
    metadata.duration_seconds_from_header = 60.0;
    metadata.begin_stream_pts_seconds_from_content = 5.0;
    metadata.end_stream_pts_seconds_from_content = 20.0;

    EXPECT_NEAR(
        metadata.get_duration_seconds(SeekMode::approximate).value(),
        60.0,
        1e-6);
  }
}

// Test that duration_seconds is calculated using average_fps_from_header and
// num_frames_from_header if duration_seconds_from_header is missing.
TEST(MetadataTest, CalculateDurationSecondsUsingFpsAndNumFrames) {
  // in approximate mode, both num_frames and fps available
  {
    StreamMetadata metadata;
    metadata.duration_seconds_from_header = std::nullopt;
    metadata.num_frames_from_header = 100;
    metadata.average_fps_from_header = 10.0;
    metadata.begin_stream_pts_seconds_from_content = std::nullopt;
    metadata.end_stream_pts_seconds_from_content = std::nullopt;

    EXPECT_NEAR(
        metadata.get_duration_seconds(SeekMode::approximate).value(),
        10.0,
        1e-6);
  }

  // in approximate mode, num_frames available but fps missing
  {
    StreamMetadata metadata;
    metadata.duration_seconds_from_header = std::nullopt;
    metadata.num_frames_from_header = 100;
    metadata.average_fps_from_header = std::nullopt;
    metadata.begin_stream_pts_seconds_from_content = std::nullopt;
    metadata.end_stream_pts_seconds_from_content = std::nullopt;

    EXPECT_EQ(
        metadata.get_duration_seconds(SeekMode::approximate), std::nullopt);
  }

  // in approximate mode, fps available but num_frames missing
  {
    StreamMetadata metadata;
    metadata.duration_seconds_from_header = std::nullopt;
    metadata.num_frames_from_header = std::nullopt;
    metadata.average_fps_from_header = 10.0;
    metadata.begin_stream_pts_seconds_from_content = std::nullopt;
    metadata.end_stream_pts_seconds_from_content = std::nullopt;

    EXPECT_EQ(
        metadata.get_duration_seconds(SeekMode::approximate), std::nullopt);
  }

  // in approximate mode, both missing
  {
    StreamMetadata metadata;
    metadata.duration_seconds_from_header = std::nullopt;
    metadata.num_frames_from_header = std::nullopt;
    metadata.average_fps_from_header = std::nullopt;
    metadata.begin_stream_pts_seconds_from_content = std::nullopt;
    metadata.end_stream_pts_seconds_from_content = std::nullopt;

    EXPECT_EQ(
        metadata.get_duration_seconds(SeekMode::approximate), std::nullopt);
  }
}

} // namespace facebook::torchcodec
