// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Demuxer.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

int read_next_packet(
    AVFormatContext* format_context,
    int active_stream_index,
    ReferenceAVPacket& packet) {
  int status = AVSUCCESS;
  do {
    status = av_read_frame(format_context, packet.get());
    if (status == AVERROR_EOF || status < AVSUCCESS) {
      return status;
    }
  } while (packet->stream_index != active_stream_index);
  return AVSUCCESS;
}

Demuxer::Demuxer(
    const std::string& file_path,
    std::optional<int> stream_index) {
  set_ffmpeg_log_level();

  AVFormatContext* raw_context = nullptr;
  int status =
      avformat_open_input(&raw_context, file_path.c_str(), nullptr, nullptr);
  STD_TORCH_CHECK(
      status == 0,
      "Could not open input file: " + file_path + " " +
          get_ffmpeg_error_string_from_error_code(status));
  STD_TORCH_CHECK(raw_context != nullptr, "Failed to allocate AVFormatContext");
  format_context_.reset(raw_context);

  status = avformat_find_stream_info(format_context_.get(), nullptr);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to find stream info: ",
      get_ffmpeg_error_string_from_error_code(status));

  active_stream_index_ = av_find_best_stream(
      format_context_.get(),
      AVMEDIA_TYPE_VIDEO,
      stream_index.value_or(-1),
      /*related_stream=*/-1,
      /*decoder_ret=*/nullptr,
      /*flags=*/0);
  STD_TORCH_CHECK(
      active_stream_index_ >= 0,
      "No valid video stream found in input file (requested index ",
      stream_index.value_or(-1),
      ").");
  stream_ = format_context_->streams[active_stream_index_];

  // We only need packets from the active stream, so tell FFmpeg to discard the
  // others. Note av_read_frame() may still return some of them under certain
  // conditions, which is why read_next_packet() also filters by stream index.
  for (unsigned int i = 0; i < format_context_->nb_streams; ++i) {
    if (i != static_cast<unsigned int>(active_stream_index_)) {
      format_context_->streams[i]->discard = AVDISCARD_ALL;
    }
  }
}

AVPacket* Demuxer::next_packet() {
  ReferenceAVPacket packet(auto_packet_);
  int status =
      read_next_packet(format_context_.get(), active_stream_index_, packet);
  if (status == AVERROR_EOF) {
    return nullptr;
  }
  STD_TORCH_CHECK(
      status >= AVSUCCESS,
      "Could not read frame from input file: ",
      get_ffmpeg_error_string_from_error_code(status));

  // Move the reference out into a fresh, independent packet the caller owns.
  // This is what makes the packet safe to hand to another thread.
  AVPacket* owned = av_packet_alloc();
  STD_TORCH_CHECK(owned != nullptr, "Failed to allocate AVPacket");
  av_packet_move_ref(owned, packet.get());
  return owned;
}

} // namespace facebook::torchcodec
