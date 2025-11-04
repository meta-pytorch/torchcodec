// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Metadata.h"

namespace facebook::torchcodec {

std::optional<double> StreamMetadata::getDurationSeconds(
    SeekMode seekMode) const {
  switch (seekMode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      // In exact mode, use the scanned content value
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
  }
  return std::nullopt;
}

double StreamMetadata::getBeginStreamSeconds(SeekMode seekMode) const {
  switch (seekMode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      return beginStreamPtsSecondsFromContent.value();
    case SeekMode::approximate:
      return 0.0;
  }
  return 0.0;
}

std::optional<double> StreamMetadata::getEndStreamSeconds(
    SeekMode seekMode) const {
  switch (seekMode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      return endStreamPtsSecondsFromContent.value();
    case SeekMode::approximate:
      return getDurationSeconds(seekMode);
  }
  return std::nullopt;
}

std::optional<int64_t> StreamMetadata::getNumFrames(SeekMode seekMode) const {
  switch (seekMode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
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
  }
  return std::nullopt;
}

std::optional<double> StreamMetadata::getAverageFps(SeekMode seekMode) const {
  switch (seekMode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      if (getNumFrames(seekMode).has_value() &&
          beginStreamPtsSecondsFromContent.has_value() &&
          endStreamPtsSecondsFromContent.has_value() &&
          (beginStreamPtsSecondsFromContent.value() !=
           endStreamPtsSecondsFromContent.value())) {
        return static_cast<double>(
            getNumFrames(seekMode).value() /
            (endStreamPtsSecondsFromContent.value() -
             beginStreamPtsSecondsFromContent.value()));
      }
      return averageFpsFromHeader;
    case SeekMode::approximate:
      return averageFpsFromHeader;
  }
  return std::nullopt;
}

} // namespace facebook::torchcodec
