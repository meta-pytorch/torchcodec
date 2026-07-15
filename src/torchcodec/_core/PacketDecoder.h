// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include "Demuxer.h"
#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

// Creates, configures and opens a codec context for `stream` using `av_codec`,
// registering the hardware device (if any) via `device_interface`. Shared by
// SingleStreamDecoder and PacketDecoder to avoid duplicating codec setup.
SharedAVCodecContext create_and_open_codec_context(
    AVStream* stream,
    const AVCodec* av_codec,
    DeviceInterface* device_interface,
    std::optional<int> thread_count);

// Decode building block: turns compressed packets into decoded (YUV) frames.
// Configured from a Demuxer's active stream; stateful. Not thread-safe.
class FORCE_PUBLIC_VISIBILITY PacketDecoder {
 public:
  explicit PacketDecoder(
      const Demuxer& demuxer,
      const StableDevice& device = StableDevice(kStableCPU),
      std::string_view device_variant = "default",
      std::optional<int> ffmpeg_thread_count = std::nullopt);

  // Feed one packet to the decoder. Borrows `packet` (does not take ownership).
  int send_packet(AVPacket* packet);
  // Signal end-of-stream so the decoder flushes its remaining frames.
  int send_eof();
  // Pull one frame. Returns AVSUCCESS with `av_frame` filled, AVERROR(EAGAIN)
  // if more input is needed, AVERROR_EOF at end, or a negative error code.
  int receive_frame(UniqueAVFrame& av_frame);

  // The stream time base, used to convert frame pts/duration to seconds.
  AVRational time_base() const {
    return time_base_;
  }

 private:
  std::unique_ptr<DeviceInterface> device_interface_;
  SharedAVCodecContext codec_context_;
  AVRational time_base_ = {};
};

} // namespace facebook::torchcodec
