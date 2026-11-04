// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "CodecParameters.h"

#include <cstring>

namespace facebook::torchcodec {

namespace {

// FF_PROFILE_UNKNOWN, AV_PROFILE_UNKNOWN and AV_LEVEL_UNKNOWN: they are all
// -99, but which of these names exist depends on the FFmpeg version.
constexpr int kUnknownProfileOrLevel = -99;

std::optional<int64_t> unless_default(
    int64_t value,
    int64_t default_value = 0) {
  if (value == default_value) {
    return std::nullopt;
  }
  return value;
}

// FFmpeg's *_name() functions return null for invalid values.
std::optional<std::string> to_optional_string(const char* name) {
  if (name == nullptr) {
    return std::nullopt;
  }
  return std::string(name);
}

std::optional<AVRational> unless_unset(const AVRational& value) {
  if (value.num == 0 || value.den == 0) {
    return std::nullopt;
  }
  return value;
}

torch::stable::Tensor to_tensor(const uint8_t* data, size_t size) {
  torch::stable::Tensor tensor =
      torch::stable::empty({static_cast<int64_t>(size)}, kStableUInt8);
  std::memcpy(tensor.mutable_data_ptr<uint8_t>(), data, size);
  return tensor;
}

// FFmpeg has no name function for these. These are the names ffprobe uses.
const char* field_order_name(AVFieldOrder field_order) {
  switch (field_order) {
    case AV_FIELD_PROGRESSIVE:
      return "progressive";
    case AV_FIELD_TT:
      return "tt";
    case AV_FIELD_BB:
      return "bb";
    case AV_FIELD_TB:
      return "tb";
    case AV_FIELD_BT:
      return "bt";
    default:
      return nullptr;
  }
}

std::optional<std::string> channel_layout_name(
    const AVCodecParameters& codecpar) {
#if FFMPEG_HAS_CH_LAYOUT
  if (codecpar.ch_layout.nb_channels == 0) {
    return std::nullopt;
  }
  std::vector<char> buffer(64);
  // Returns the size the description needs, which may be more than we gave.
  int size = av_channel_layout_describe(
      &codecpar.ch_layout, buffer.data(), buffer.size());
  if (size <= 0) {
    return std::nullopt;
  }
  if (static_cast<size_t>(size) > buffer.size()) {
    buffer.resize(size);
    av_channel_layout_describe(
        &codecpar.ch_layout, buffer.data(), buffer.size());
  }
  return std::string(buffer.data());
#else
  if (codecpar.channels == 0) {
    return std::nullopt;
  }
  char buffer[256];
  av_get_channel_layout_string(
      buffer, sizeof(buffer), codecpar.channels, codecpar.channel_layout);
  return std::string(buffer);
#endif
}

// Coded side data is only on AVCodecParameters from FFmpeg 6.1. Before that it
// was on AVStream, where we only read the display matrix, as `rotation`.
void set_coded_side_data(
    [[maybe_unused]] CodecParameters& params,
    [[maybe_unused]] const AVCodecParameters& codecpar) {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(60, 31, 100)
  for (int i = 0; i < codecpar.nb_coded_side_data; ++i) {
    const AVPacketSideData& side_data = codecpar.coded_side_data[i];
    const char* name = av_packet_side_data_name(side_data.type);
    // A type this FFmpeg can't name is a type it doesn't know what to do with
    // either, so there's nothing a decoder would read from it.
    if (name != nullptr) {
      params.coded_side_data_names.emplace_back(name);
      params.coded_side_data_values.push_back(
          to_tensor(side_data.data, side_data.size));
    }
  }
#endif
}

void set_video_fields(CodecParameters& params, const AVStream& av_stream) {
  const AVCodecParameters& codecpar = *av_stream.codecpar;
  params.width = unless_default(codecpar.width);
  params.height = unless_default(codecpar.height);
  params.pixel_format = to_optional_string(
      av_get_pix_fmt_name(static_cast<AVPixelFormat>(codecpar.format)));
  params.sample_aspect_ratio = unless_unset(codecpar.sample_aspect_ratio);
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(60, 11, 100)
  params.framerate = unless_unset(codecpar.framerate);
#endif
  params.r_frame_rate = unless_unset(av_stream.r_frame_rate);
  params.field_order =
      to_optional_string(field_order_name(codecpar.field_order));
  if (codecpar.color_range != AVCOL_RANGE_UNSPECIFIED) {
    params.color_range =
        to_optional_string(av_color_range_name(codecpar.color_range));
  }
  if (codecpar.color_primaries != AVCOL_PRI_UNSPECIFIED) {
    params.color_primaries =
        to_optional_string(av_color_primaries_name(codecpar.color_primaries));
  }
  if (codecpar.color_trc != AVCOL_TRC_UNSPECIFIED) {
    params.color_transfer_characteristic =
        to_optional_string(av_color_transfer_name(codecpar.color_trc));
  }
  if (codecpar.color_space != AVCOL_SPC_UNSPECIFIED) {
    params.color_space =
        to_optional_string(av_color_space_name(codecpar.color_space));
  }
  if (codecpar.chroma_location != AVCHROMA_LOC_UNSPECIFIED) {
    params.chroma_location =
        to_optional_string(av_chroma_location_name(codecpar.chroma_location));
  }
  params.video_delay = unless_default(codecpar.video_delay);
  params.rotation = get_rotation_from_stream(&av_stream);
}

void set_audio_fields(
    CodecParameters& params,
    const AVCodecParameters& codecpar) {
  params.sample_format = to_optional_string(
      av_get_sample_fmt_name(static_cast<AVSampleFormat>(codecpar.format)));
  params.sample_rate = unless_default(codecpar.sample_rate);
  params.channel_layout = channel_layout_name(codecpar);
  params.block_align = unless_default(codecpar.block_align);
  params.frame_size = unless_default(codecpar.frame_size);
  params.initial_padding = unless_default(codecpar.initial_padding);
  params.trailing_padding = unless_default(codecpar.trailing_padding);
  params.seek_preroll = unless_default(codecpar.seek_preroll);
}

} // namespace

CodecParameters get_codec_parameters(const AVStream& av_stream) {
  const AVCodecParameters& codecpar = *av_stream.codecpar;

  CodecParameters params;
  params.codec = avcodec_get_name(codecpar.codec_id);
  params.time_base = av_stream.time_base;
  if (codecpar.extradata_size > 0) {
    params.extradata = to_tensor(codecpar.extradata, codecpar.extradata_size);
  }
  params.codec_tag = unless_default(codecpar.codec_tag);
  params.profile = unless_default(codecpar.profile, kUnknownProfileOrLevel);
  params.level = unless_default(codecpar.level, kUnknownProfileOrLevel);
  params.bit_rate = unless_default(codecpar.bit_rate);
  params.bits_per_coded_sample = unless_default(codecpar.bits_per_coded_sample);
  params.bits_per_raw_sample = unless_default(codecpar.bits_per_raw_sample);
  set_coded_side_data(params, codecpar);

  if (codecpar.codec_type == AVMEDIA_TYPE_VIDEO) {
    set_video_fields(params, av_stream);
  } else if (codecpar.codec_type == AVMEDIA_TYPE_AUDIO) {
    set_audio_fields(params, codecpar);
  }
  return params;
}

} // namespace facebook::torchcodec
