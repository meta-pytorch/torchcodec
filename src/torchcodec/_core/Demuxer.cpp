// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Demuxer.h"

#include <algorithm>
#include <sstream>

#include "StableABICompat.h"

namespace facebook::torchcodec {

// FFmpeg reports "this seek cannot be performed" as a bare -1, i.e. EPERM,
// which renders as the very misleading "Operation not permitted". It covers
// both a target that the demuxer can't reach and a demuxer with no seeking
// support whatsoever.
std::string get_seek_error_message(
    const AVFormatContext* format_context,
    int64_t desired_pts,
    int status) {
  std::stringstream ss;
  ss << "Could not seek file to pts=" << desired_pts << ": "
     << get_ffmpeg_error_string_from_error_code(status) << ".";
  if (status == AVERROR(EPERM)) {
    ss << " This is either because that timestamp is out of range, or because"
       << " the '" << format_context->iformat->name << "' format does not"
       << " support seeking.";
  }
  return ss.str();
}

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

namespace {
// "video" / "audio", for error messages.
const char* printable(AVMediaType media_type) {
  const char* name = av_get_media_type_string(media_type);
  return name == nullptr ? "unknown" : name;
}
} // namespace

Demuxer::Demuxer(
    const std::string& file_path,
    std::optional<int> stream_index,
    AVMediaType media_type)
    : media_type_(media_type) {
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

  select_stream(stream_index);
}

Demuxer::Demuxer(
    std::unique_ptr<AVIOContextHolder> avio_context_holder,
    std::optional<int> stream_index,
    AVMediaType media_type)
    : avio_context_holder_(std::move(avio_context_holder)),
      media_type_(media_type) {
  set_ffmpeg_log_level();

  STD_TORCH_CHECK(avio_context_holder_ != nullptr, "Context holder is null");

  // FFmpeg takes a reference to the pointer in the call to open, so we can't
  // hand it a unique_ptr. That means we must free the context ourselves if the
  // open fails.
  AVFormatContext* raw_context = avformat_alloc_context();
  STD_TORCH_CHECK(raw_context != nullptr, "Failed to allocate AVFormatContext");
  raw_context->pb = avio_context_holder_->get_avio_context();

  int status = avformat_open_input(&raw_context, nullptr, nullptr, nullptr);
  if (status != 0) {
    avformat_free_context(raw_context);
    STD_TORCH_CHECK(
        false,
        "Could not open input buffer: " +
            get_ffmpeg_error_string_from_error_code(status));
  }
  format_context_.reset(raw_context);

  select_stream(stream_index);
}

void Demuxer::validate_requested_stream(int stream_index) {
  int num_streams = static_cast<int>(format_context_->nb_streams);
  STD_TORCH_CHECK(
      stream_index >= 0 && stream_index < num_streams,
      "The stream index ",
      stream_index,
      " is not a valid stream. The file has ",
      num_streams,
      " streams, so the index must be in [0, ",
      num_streams - 1,
      "].");

  AVMediaType stream_media_type =
      format_context_->streams[stream_index]->codecpar->codec_type;
  STD_TORCH_CHECK(
      stream_media_type == media_type_,
      "The stream at index ",
      stream_index,
      " is not a ",
      printable(media_type_),
      " stream, it is of type '",
      printable(stream_media_type),
      "'.");
}

void Demuxer::select_stream(std::optional<int> stream_index) {
  int status = avformat_find_stream_info(format_context_.get(), nullptr);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to find stream info: ",
      get_ffmpeg_error_string_from_error_code(status));

  if (stream_index.has_value()) {
    validate_requested_stream(*stream_index);
  }

  active_stream_index_ = av_find_best_stream(
      format_context_.get(),
      media_type_,
      stream_index.value_or(-1),
      /*related_stream=*/-1,
      /*decoder_ret=*/nullptr,
      /*flags=*/0);
  STD_TORCH_CHECK(
      active_stream_index_ >= 0,
      "No valid ",
      printable(media_type_),
      " stream found in input file.");
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

void Demuxer::seek(double seconds) {
  int64_t desired_pts = seconds_to_closest_pts(seconds, stream_->time_base);

  int status = avformat_seek_file(
      format_context_.get(),
      active_stream_index_,
      INT64_MIN,
      desired_pts,
      desired_pts,
      0);
  STD_TORCH_CHECK(
      status >= 0,
      get_seek_error_message(format_context_.get(), desired_pts, status));
}

namespace {
struct ScannedPacket {
  int64_t pts;
  int64_t duration;
  bool is_key_frame;
};
} // namespace

StreamIndex Demuxer::scan() {
  std::vector<ScannedPacket> packets;
  if (stream_->nb_frames > 0) {
    packets.reserve(static_cast<size_t>(stream_->nb_frames));
  }

  seek(0);

  while (true) {
    ReferenceAVPacket packet(auto_packet_);
    int status =
        read_next_packet(format_context_.get(), active_stream_index_, packet);
    if (status == AVERROR_EOF) {
      break;
    }
    STD_TORCH_CHECK(
        status >= AVSUCCESS,
        "Could not read frame from input file: ",
        get_ffmpeg_error_string_from_error_code(status));

    if (packet->flags & AV_PKT_FLAG_DISCARD) {
      continue;
    }

    packets.push_back(
        {get_pts_or_dts(packet),
         packet->duration,
         (packet->flags & AV_PKT_FLAG_KEY) != 0});
  }

  std::stable_sort(
      packets.begin(),
      packets.end(),
      [](const ScannedPacket& a, const ScannedPacket& b) {
        return a.pts < b.pts;
      });

  seek(0);

  auto num_frames = static_cast<int64_t>(packets.size());
  StreamIndex index{
      torch::stable::empty({num_frames}, kStableInt64),
      torch::stable::empty({num_frames}, kStableInt64),
      torch::stable::empty({num_frames}, kStableBool),
      stream_->time_base};

  auto pts = mutable_accessor<int64_t, 1>(index.pts);
  auto duration = mutable_accessor<int64_t, 1>(index.duration);
  auto is_key_frame = mutable_accessor<bool, 1>(index.is_key_frame);
  for (int64_t i = 0; i < num_frames; ++i) {
    pts[i] = packets[i].pts;
    duration[i] = packets[i].duration;
    is_key_frame[i] = packets[i].is_key_frame;
  }

  return index;
}

UniqueAVPacket Demuxer::next_packet() {
  // TODO_API_BREAKDOWN CC P2: Not a fan of the ReferenceAVPacket / AutoAVPacket
  // / UniqueAVPacket dance here. Can we simplify?
  ReferenceAVPacket packet(auto_packet_);
  int status =
      read_next_packet(format_context_.get(), active_stream_index_, packet);
  if (status == AVERROR_EOF) {
    return UniqueAVPacket{};
  }
  STD_TORCH_CHECK(
      status >= AVSUCCESS,
      "Could not read frame from input file: ",
      get_ffmpeg_error_string_from_error_code(status));

  // Move the reference out into a fresh, independent packet the caller owns.
  // This is what makes the packet safe to hand to another thread.
  UniqueAVPacket owned(av_packet_alloc());
  STD_TORCH_CHECK(owned != nullptr, "Failed to allocate AVPacket");
  av_packet_move_ref(owned.get(), packet.get());
  return owned;
}

} // namespace facebook::torchcodec
