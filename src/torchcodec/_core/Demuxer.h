// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "AVIOContextHolder.h"
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

std::string get_seek_error_message(
    const AVFormatContext* format_context,
    int64_t desired_pts,
    int status);

// The frames of the active stream, in presentation order: three parallel
// tensors of length N, plus the time base `pts` and `duration` are expressed
// in.
struct StreamIndex {
  torch::stable::Tensor pts; // int64 [N]
  torch::stable::Tensor duration; // int64 [N]
  torch::stable::Tensor is_key_frame; // bool [N]
  AVRational time_base;
};

// Demux building block: owns an AVFormatContext, selects one video stream, and
// yields its (compressed) packets. Does no decoding. Not thread-safe.
class FORCE_PUBLIC_VISIBILITY Demuxer {
 public:
  explicit Demuxer(
      const std::string& file_path,
      std::optional<int> stream_index = std::nullopt);

  explicit Demuxer(
      std::unique_ptr<AVIOContextHolder> avio_context_holder,
      std::optional<int> stream_index = std::nullopt);

  // Returns the next packet for the active stream as a freshly-allocated
  // packet, or a null packet at end of stream.
  UniqueAVPacket next_packet();

  void seek(double seconds);

  // Demuxes the entire stream, without decoding, and returns one entry per
  // frame sorted by pts. Leaves the demuxer back at the start of the stream,
  // and keeps no state of its own.
  StreamIndex scan();

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
  void validate_requested_stream(int stream_index);
  void select_stream(std::optional<int> stream_index);

  // Declared before format_context_ so that it outlives it: the format context
  // reads through the AVIOContext this holds.
  std::unique_ptr<AVIOContextHolder> avio_context_holder_;
  UniqueDecodingAVFormatContext format_context_;
  int active_stream_index_ = -1;
  AVStream* stream_ = nullptr;
  AutoAVPacket auto_packet_;
};

} // namespace facebook::torchcodec
