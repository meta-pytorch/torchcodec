// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
      std::optional<int> ffmpeg_thread_count = std::nullopt);

  // Feed one packet to the decoder. Borrows `packet` (does not take ownership).
  int send_packet(AVPacket* packet);
  // Signal end-of-stream so the decoder flushes its remaining frames.
  int send_eof();
  // Pull one frame. Returns AVSUCCESS with `av_frame` filled, AVERROR(EAGAIN)
  // if more input is needed, AVERROR_EOF at end, or a negative error code.
  int receive_frame(UniqueAVFrame& av_frame);

  // Whether the frame's pixel data is on this decoder's device or in host
  // memory. A CUDA decoder yields host frames for streams NVDEC can't handle.
  bool is_device_frame(const UniqueAVFrame& av_frame) const {
    return device_interface_->is_device_frame(av_frame);
  }

  // The device this decoder decodes on. Frames may still be in host memory (see
  // is_device_frame).
  const StableDevice& device() const {
    return device_interface_->device();
  }

  // The stream time base, used to convert frame pts/duration to seconds.
  AVRational time_base() const {
    return time_base_;
  }

 private:
  std::unique_ptr<DeviceInterface> device_interface_;
  SharedAVCodecContext codec_context_;
  AVRational time_base_ = {};
};

// A decoded frame's own samples, before any color conversion.
struct FramePlanes {
  // One view per component, in the frame's native order: (Y, U, V) for YUV,
  // (R, G, B) for RGB codecs, (Y,) for grayscale, plus a trailing alpha view
  // when the format has one.
  std::vector<torch::stable::Tensor> planes;
  std::string pix_fmt; // FFmpeg pixel-format name, e.g. "yuv420p"
  int64_t colorspace = 0; // AVColorSpace
  int64_t color_range = 0; // AVColorRange
};

// Zero-copy views of a decoded frame's own samples, one per component, before
// any color conversion. Each view is a strided window over the frame's memory
// as described by the pixel format's per-component plane/offset/step, so
// semi-planar (nv12, p016) and packed (yuyv422, bgra...) layouts come back as
// clean separate components without a copy -- the interleaving lives in the
// sample stride. 10-/12-bit samples come back as uint16 views.
//
// `device` is where the frame's samples live (its NVDEC surface for GPU frames,
// host memory otherwise); the views are built on it, so no data is moved.
//
// `frame_owner` is whatever owns `av_frame`'s lifetime; each view captures it
// so the samples stay valid after the caller drops the frame they came from.
// Note that we can't instead take a reference on the AVFrame itself: GPU frames
// aren't refcounted (their data points into a torch tensor attached as
// opaque_ref), so av_frame_ref() would deep-copy them as if they were host
// memory.
FORCE_PUBLIC_VISIBILITY FramePlanes frame_to_planes(
    const AVFrame& av_frame,
    const StableDevice& device,
    const torch::stable::Tensor& frame_owner);

} // namespace facebook::torchcodec
