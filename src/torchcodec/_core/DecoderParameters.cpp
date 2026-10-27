// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecoderParameters.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <string_view>

namespace facebook::torchcodec {

DecoderParameters::DecoderParameters()
    : codec_parameters(avcodec_parameters_alloc()) {
  STD_TORCH_CHECK(
      codec_parameters != nullptr, "Failed to allocate codec parameters");
}

DecoderParameters get_decoder_parameters(
    const AVFormatContext& format_context,
    int stream_index) {
  const AVStream* stream = format_context.streams[stream_index];

  DecoderParameters parameters;
  int ret = avcodec_parameters_copy(
      parameters.codec_parameters.get(), stream->codecpar);
  STD_TORCH_CHECK(
      ret >= AVSUCCESS,
      "Failed to copy codec parameters: ",
      get_ffmpeg_error_string_from_error_code(ret));
  parameters.time_base = stream->time_base;
  parameters.r_frame_rate = stream->r_frame_rate;
  if (const int32_t* matrix = get_display_matrix_from_stream(stream)) {
    parameters.display_matrix.emplace();
    std::copy(
        matrix,
        matrix + parameters.display_matrix->size(),
        parameters.display_matrix->begin());
  }
  parameters.packets_may_be_misaligned_after_reset =
      std::string_view(format_context.iformat->name) == "mpeg";
  return parameters;
}

namespace {

const std::string kCodedSideDataPrefix = "coded_side_data:";

std::string bytes_to_hex(const uint8_t* data, int size) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(2 * static_cast<size_t>(size));
  for (int i = 0; i < size; ++i) {
    hex += kDigits[data[i] >> 4];
    hex += kDigits[data[i] & 0xf];
  }
  return hex;
}

std::string rational_to_string(AVRational value) {
  return std::to_string(value.num) + "/" + std::to_string(value.den);
}

// FFmpeg's names for an enum value, falling back to the plain integer for the
// values FFmpeg has no name for (e.g. reserved color primaries), so that every
// value round-trips.
std::string enum_to_string(int value, const char* name) {
  return name != nullptr ? std::string(name) : std::to_string(value);
}

constexpr std::array<const char*, 6> kFieldOrderNames =
    {"unknown", "progressive", "tt", "bb", "tb", "bt"};

const char* field_order_name(int field_order) {
  return field_order >= 0 &&
          field_order < static_cast<int>(kFieldOrderNames.size())
      ? kFieldOrderNames[field_order]
      : nullptr;
}

std::string channel_layout_to_string(
    const AVCodecParameters& codec_parameters) {
  char buffer[256] = {0};
#if FFMPEG_HAS_CH_LAYOUT
  av_channel_layout_describe(
      &codec_parameters.ch_layout, buffer, sizeof(buffer));
#else
  av_get_channel_layout_string(
      buffer,
      sizeof(buffer),
      codec_parameters.channels,
      codec_parameters.channel_layout);
#endif
  return buffer;
}

} // namespace

std::map<std::string, std::string> decoder_parameters_to_strings(
    const DecoderParameters& parameters) {
  const AVCodecParameters& par = *parameters.codec_parameters;
  std::map<std::string, std::string> values;
  bool is_video = par.codec_type == AVMEDIA_TYPE_VIDEO;

  values["codec"] = avcodec_get_name(par.codec_id);
  values["time_base"] = rational_to_string(parameters.time_base);
  values["codec_tag"] = std::to_string(par.codec_tag);
  values["profile"] = std::to_string(par.profile);
  values["level"] = std::to_string(par.level);
  values["bit_rate"] = std::to_string(par.bit_rate);
  values["bits_per_coded_sample"] = std::to_string(par.bits_per_coded_sample);
  values["bits_per_raw_sample"] = std::to_string(par.bits_per_raw_sample);
  values["packets_may_be_misaligned_after_reset"] =
      parameters.packets_may_be_misaligned_after_reset ? "1" : "0";

  if (is_video) {
    values["width"] = std::to_string(par.width);
    values["height"] = std::to_string(par.height);
    if (par.format >= 0) {
      values["pixel_format"] = enum_to_string(
          par.format,
          av_get_pix_fmt_name(static_cast<AVPixelFormat>(par.format)));
    }
    if (par.sample_aspect_ratio.num > 0 && par.sample_aspect_ratio.den > 0) {
      values["sample_aspect_ratio"] =
          rational_to_string(par.sample_aspect_ratio);
    }
#if LIBAVCODEC_VERSION_MAJOR >= 60
    if (par.framerate.num > 0 && par.framerate.den > 0) {
      values["framerate"] = rational_to_string(par.framerate);
    }
#endif
    if (parameters.r_frame_rate.num > 0 && parameters.r_frame_rate.den > 0) {
      values["r_frame_rate"] = rational_to_string(parameters.r_frame_rate);
    }
    values["field_order"] =
        enum_to_string(par.field_order, field_order_name(par.field_order));
    values["color_range"] =
        enum_to_string(par.color_range, av_color_range_name(par.color_range));
    values["color_primaries"] = enum_to_string(
        par.color_primaries, av_color_primaries_name(par.color_primaries));
    values["color_transfer_characteristic"] =
        enum_to_string(par.color_trc, av_color_transfer_name(par.color_trc));
    values["color_space"] =
        enum_to_string(par.color_space, av_color_space_name(par.color_space));
    values["chroma_location"] = enum_to_string(
        par.chroma_location, av_chroma_location_name(par.chroma_location));
    values["video_delay"] = std::to_string(par.video_delay);
    if (parameters.display_matrix.has_value()) {
      double rotation =
          av_display_rotation_get(parameters.display_matrix->data());
      if (!std::isnan(rotation)) {
        std::ostringstream stream;
        stream.precision(17);
        stream << rotation;
        values["rotation"] = stream.str();
      }
    }
  } else {
    if (par.format >= 0) {
      values["sample_format"] = enum_to_string(
          par.format,
          av_get_sample_fmt_name(static_cast<AVSampleFormat>(par.format)));
    }
    values["sample_rate"] = std::to_string(par.sample_rate);
    if (get_num_channels(&par) > 0) {
      values["channel_layout"] = channel_layout_to_string(par);
    }
    values["block_align"] = std::to_string(par.block_align);
    values["frame_size"] = std::to_string(par.frame_size);
    values["initial_padding"] = std::to_string(par.initial_padding);
    values["trailing_padding"] = std::to_string(par.trailing_padding);
    values["seek_preroll"] = std::to_string(par.seek_preroll);
  }

  if (par.extradata_size > 0) {
    values["extradata"] = bytes_to_hex(par.extradata, par.extradata_size);
  }

#if FFMPEG_HAS_CODED_SIDE_DATA
  for (int i = 0; i < par.nb_coded_side_data; ++i) {
    const AVPacketSideData& side_data = par.coded_side_data[i];
    // That one is what `rotation` describes.
    if (side_data.type == AV_PKT_DATA_DISPLAYMATRIX) {
      continue;
    }
    const char* name = av_packet_side_data_name(side_data.type);
    if (name == nullptr) {
      continue;
    }
    values[kCodedSideDataPrefix + name] =
        bytes_to_hex(side_data.data, side_data.size);
  }
#endif
  return values;
}

} // namespace facebook::torchcodec
