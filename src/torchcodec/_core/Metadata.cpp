// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Metadata.h"
#include "torch/types.h"

namespace facebook::torchcodec {

std::optional<double> StreamMetadata::getDurationSeconds(
    SeekMode seekMode) const {
  switch (seekMode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      TORCH_CHECK(
          endStreamPtsSecondsFromContent.has_value() &&
              beginStreamPtsSecondsFromContent.has_value(),
          "Missing beginStreamPtsSecondsFromContent or endStreamPtsSecondsFromContent");
      return endStreamPtsSecondsFromContent.value() -
          beginStreamPtsSecondsFromContent.value();
    case SeekMode::approximate:
      if (durationSecondsFromHeader.has_value()) {
        return durationSecondsFromHeader.value();
      }
      if (numFramesFromHeader.has_value() && averageFpsFromHeader.has_value() &&
          averageFpsFromHeader.value() != 0.0) {
        return static_cast<double>(numFramesFromHeader.value()) /
            averageFpsFromHeader.value();
      }
      return std::nullopt;
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

double StreamMetadata::getBeginStreamSeconds(SeekMode seekMode) const {
  switch (seekMode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      TORCH_CHECK(
          beginStreamPtsSecondsFromContent.has_value(),
          "Missing beginStreamPtsSecondsFromContent");
      return beginStreamPtsSecondsFromContent.value();
    case SeekMode::approximate:
      if (beginStreamPtsSecondsFromContent.has_value()) {
        return beginStreamPtsSecondsFromContent.value();
      }
      return 0.0;
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

std::optional<double> StreamMetadata::getEndStreamSeconds(
    SeekMode seekMode) const {
  switch (seekMode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      TORCH_CHECK(
          endStreamPtsSecondsFromContent.has_value(),
          "Missing endStreamPtsSecondsFromContent");
      return endStreamPtsSecondsFromContent.value();
    case SeekMode::approximate:
      if (endStreamPtsSecondsFromContent.has_value()) {
        return endStreamPtsSecondsFromContent.value();
      }
      return getDurationSeconds(seekMode);
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

std::optional<int64_t> StreamMetadata::getNumFrames(SeekMode seekMode) const {
  switch (seekMode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      TORCH_CHECK(
          numFramesFromContent.has_value(), "Missing numFramesFromContent");
      return numFramesFromContent.value();
    case SeekMode::approximate: {
      if (numFramesFromHeader.has_value()) {
        return numFramesFromHeader.value();
      }
      if (averageFpsFromHeader.has_value() &&
          durationSecondsFromHeader.has_value()) {
        return static_cast<int64_t>(
            averageFpsFromHeader.value() * durationSecondsFromHeader.value());
      }
      return std::nullopt;
    }
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

std::optional<double> StreamMetadata::getAverageFps(SeekMode seekMode) const {
  switch (seekMode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact: {
      auto numFrames = getNumFrames(seekMode);
      if (numFrames.has_value() &&
          beginStreamPtsSecondsFromContent.has_value() &&
          endStreamPtsSecondsFromContent.has_value()) {
        double duration = endStreamPtsSecondsFromContent.value() -
            beginStreamPtsSecondsFromContent.value();
        if (duration != 0.0) {
          return static_cast<double>(numFrames.value()) / duration;
        }
      }
      return averageFpsFromHeader;
    }
    case SeekMode::approximate:
      return averageFpsFromHeader;
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

} // namespace facebook::torchcodec
