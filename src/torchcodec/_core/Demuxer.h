// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string>

#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

// Reads the next packet belonging to active_stream_index from format_context
// into `packet`. Returns AVSUCCESS with `packet` filled, AVERROR_EOF at end of
// stream, or a negative error code otherwise. Shared by Demuxer and
// SingleStreamDecoder so the demux + stream-filter loop lives in one place.
int read_next_packet(
    AVFormatContext* format_context,
    int active_stream_index,
    ReferenceAVPacket& packet);

// Demux building block: owns an AVFormatContext, selects one video stream, and
// yields its (compressed) packets. Does no decoding. Not thread-safe.
class FORCE_PUBLIC_VISIBILITY Demuxer {
 public:
  explicit Demuxer(
      const std::string& file_path,
      std::optional<int> stream_index = std::nullopt);

  // Returns the next packet for the active stream as a freshly-allocated,
  // owning AVPacket (the caller takes ownership and must av_packet_free it), or
  // nullptr at end of stream.
  AVPacket* next_packet();

  AVStream* active_stream() const {
    return stream_;
  }

  int active_stream_index() const {
    return active_stream_index_;
  }

  const UniqueDecodingAVFormatContext& format_context() const {
    return format_context_;
  }

 private:
  UniqueDecodingAVFormatContext format_context_;
  int active_stream_index_ = -1;
  AVStream* stream_ = nullptr;
  AutoAVPacket auto_packet_;
};

} // namespace facebook::torchcodec
