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

// FFmpeg reports "this seek cannot be performed" as a bare -1, i.e. EPERM,
// which renders as the very misleading "Operation not permitted". It covers
// both a target that the demuxer can't reach and a demuxer with no seeking
// support whatsoever.
std::string get_seek_error_message(
    const AVFormatContext* format_context,
    int64_t desired_pts,
    int status);

// Demux building block: owns an AVFormatContext, selects one video stream, and
// yields its (compressed) packets. Does no decoding. Not thread-safe.
class FORCE_PUBLIC_VISIBILITY Demuxer {
 public:
  explicit Demuxer(
      const std::string& file_path,
      std::optional<int> stream_index = std::nullopt);

  // Returns the next packet for the active stream as a freshly-allocated
  // packet, or a null packet at end of stream.
  UniqueAVPacket next_packet();

  // Seeks so that the next packet read is a keyframe's, close to `seconds`.
  // Same semantics as SingleStreamDecoder's approximate seek mode: we hand
  // FFmpeg the target and take whatever keyframe it lands on. That is usually
  // the keyframe preceding `seconds`, but on streams whose keyframes are
  // reordered it can be one displayed *after* it, because the container's
  // index is in decode order (see https://trac.ffmpeg.org/ticket/11137).
  // Landing exactly needs a presentation-order index of our own, which is what
  // SingleStreamDecoder's exact mode scans for.
  //
  // This deliberately does not touch any decoder: a PacketDecoder's reference
  // frames are stale after a seek and it is the caller's job to reset() it,
  // because the decoder may well live on another thread.
  void seek(double seconds);

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
