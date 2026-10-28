// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecoderParameters.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
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

namespace {

std::vector<uint8_t> hex_to_bytes(
    const std::string& key,
    const std::string& hex) {
  auto digit = [&](char c) -> uint8_t {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
      return c - 'a' + 10;
    }
    STD_TORCH_CHECK(false, "Invalid value for ", key, ": expected hex digits.");
  };
  STD_TORCH_CHECK(
      hex.size() % 2 == 0, "Invalid value for ", key, ": expected hex digits.");
  std::vector<uint8_t> bytes(hex.size() / 2);
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = (digit(hex[2 * i]) << 4) | digit(hex[2 * i + 1]);
  }
  return bytes;
}

AVRational parse_rational(const std::string& key, const std::string& value) {
  int num = 0;
  int den = 0;
  char slash = 0;
  std::istringstream stream(value);
  stream >> num >> slash >> den;
  STD_TORCH_CHECK(
      !stream.fail() && stream.eof() && slash == '/' && den != 0,
      "Invalid value for ",
      key,
      ": expected a rational num/den, got '",
      value,
      "'.");
  return AVRational{num, den};
}

int64_t parse_int(const std::string& key, const std::string& value) {
  size_t end = 0;
  int64_t result = 0;
  try {
    result = std::stoll(value, &end);
  } catch (const std::exception&) {
    end = 0;
  }
  STD_TORCH_CHECK(
      end != 0 && end == value.size(),
      "Invalid value for ",
      key,
      ": expected an integer, got '",
      value,
      "'.");
  return result;
}

int parse_int32(const std::string& key, const std::string& value) {
  int64_t result = parse_int(key, value);
  STD_TORCH_CHECK(
      result >= std::numeric_limits<int>::min() &&
          result <= std::numeric_limits<int>::max(),
      "Invalid value for ",
      key,
      ": ",
      value,
      " is out of range.");
  return static_cast<int>(result);
}

int parse_enum(
    const std::string& key,
    const std::string& value,
    int (*from_name)(const char*)) {
  int result = from_name(value.c_str());
  if (result >= 0) {
    return result;
  }
  // Not a name: it may be one of the integers enum_to_string() falls back to.
  size_t end = 0;
  try {
    result = std::stoi(value, &end);
  } catch (const std::exception&) {
    end = 0;
  }
  STD_TORCH_CHECK(
      end != 0 && end == value.size() && result >= 0,
      "Invalid value for ",
      key,
      ": '",
      value,
      "' isn't a name FFmpeg knows.");
  return result;
}

int field_order_from_name(const char* name) {
  for (size_t i = 0; i < kFieldOrderNames.size(); ++i) {
    if (std::string_view(name) == kFieldOrderNames[i]) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

int pixel_format_from_name(const char* name) {
  return av_get_pix_fmt(name);
}

int sample_format_from_name(const char* name) {
  return av_get_sample_fmt(name);
}

void set_channel_layout(
    AVCodecParameters& codec_parameters,
    const std::string& value) {
#if FFMPEG_HAS_CH_LAYOUT
  av_channel_layout_uninit(&codec_parameters.ch_layout);
  int status =
      av_channel_layout_from_string(&codec_parameters.ch_layout, value.c_str());
  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "Invalid value for channel_layout: '",
      value,
      "' isn't a channel layout FFmpeg knows.");
#else
  uint64_t layout = 0;
  int num_channels = 0;
  int status =
      av_get_extended_channel_layout(value.c_str(), &layout, &num_channels);
  STD_TORCH_CHECK(
      status >= 0,
      "Invalid value for channel_layout: '",
      value,
      "' isn't a channel layout FFmpeg knows.");
  codec_parameters.channel_layout = layout;
  codec_parameters.channels = num_channels;
#endif
}

} // namespace

DecoderParameters decoder_parameters_from_strings(
    const std::map<std::string, std::string>& values) {
  DecoderParameters parameters;
  AVCodecParameters& par = *parameters.codec_parameters;

  auto get = [&](const std::string& key) -> const std::string& {
    auto it = values.find(key);
    STD_TORCH_CHECK(it != values.end(), key, " is required.");
    return it->second;
  };

  const std::string& media_type = get("media_type");
  STD_TORCH_CHECK(
      media_type == "video" || media_type == "audio",
      "Invalid media_type: '",
      media_type,
      "'.");
  bool is_video = media_type == "video";
  par.codec_type = is_video ? AVMEDIA_TYPE_VIDEO : AVMEDIA_TYPE_AUDIO;

  const std::string& codec = get("codec");
  const AVCodecDescriptor* descriptor =
      avcodec_descriptor_get_by_name(codec.c_str());
  STD_TORCH_CHECK(
      descriptor != nullptr,
      "Unknown codec: '",
      codec,
      "'. Expected one of the names FFmpeg's `ffmpeg -codecs` lists.");
  STD_TORCH_CHECK(
      descriptor->type == par.codec_type,
      "'",
      codec,
      "' is not a",
      is_video ? " video" : "n audio",
      " codec.");
  par.codec_id = descriptor->id;

  parameters.time_base = parse_rational("time_base", get("time_base"));
  STD_TORCH_CHECK(
      parameters.time_base.num > 0 && parameters.time_base.den > 0,
      "time_base must be positive.");

  for (const auto& [key, value] : values) {
    if (key == "media_type" || key == "codec" || key == "time_base") {
      continue;
    } else if (key == "extradata") {
      std::vector<uint8_t> extradata = hex_to_bytes(key, value);
      par.extradata = static_cast<uint8_t*>(
          av_mallocz(extradata.size() + AV_INPUT_BUFFER_PADDING_SIZE));
      STD_TORCH_CHECK(par.extradata != nullptr, "Failed to allocate extradata");
      std::memcpy(par.extradata, extradata.data(), extradata.size());
      par.extradata_size = static_cast<int>(extradata.size());
    } else if (key.rfind(kCodedSideDataPrefix, 0) == 0) {
#if FFMPEG_HAS_CODED_SIDE_DATA
      std::vector<uint8_t> payload = hex_to_bytes(key, value);
      AVPacketSideData* side_data = av_packet_side_data_new(
          &par.coded_side_data,
          &par.nb_coded_side_data,
          side_data_type_from_name(key.substr(kCodedSideDataPrefix.size())),
          payload.size(),
          0);
      STD_TORCH_CHECK(side_data != nullptr, "Failed to allocate side data");
      std::memcpy(side_data->data, payload.data(), payload.size());
#else
      STD_TORCH_CHECK(false, "coded_side_data requires FFmpeg 6.1 or above.");
#endif
    } else if (key == "packets_may_be_misaligned_after_reset") {
      parameters.packets_may_be_misaligned_after_reset =
          parse_int(key, value) != 0;
    } else if (key == "codec_tag") {
      int64_t codec_tag = parse_int(key, value);
      STD_TORCH_CHECK(
          codec_tag >= 0 && codec_tag <= std::numeric_limits<uint32_t>::max(),
          "Invalid value for codec_tag: ",
          value,
          " is out of range.");
      par.codec_tag = static_cast<uint32_t>(codec_tag);
    } else if (key == "profile") {
      par.profile = parse_int32(key, value);
    } else if (key == "level") {
      par.level = parse_int32(key, value);
    } else if (key == "bit_rate") {
      par.bit_rate = parse_int(key, value);
    } else if (key == "bits_per_coded_sample") {
      par.bits_per_coded_sample = parse_int32(key, value);
    } else if (key == "bits_per_raw_sample") {
      par.bits_per_raw_sample = parse_int32(key, value);
    } else if (is_video && key == "width") {
      par.width = parse_int32(key, value);
    } else if (is_video && key == "height") {
      par.height = parse_int32(key, value);
    } else if (is_video && key == "pixel_format") {
      par.format = parse_enum(key, value, pixel_format_from_name);
    } else if (is_video && key == "sample_aspect_ratio") {
      par.sample_aspect_ratio = parse_rational(key, value);
    } else if (is_video && key == "framerate") {
#if LIBAVCODEC_VERSION_MAJOR >= 60
      par.framerate = parse_rational(key, value);
#else
      STD_TORCH_CHECK(false, "framerate requires FFmpeg 6 or above.");
#endif
    } else if (is_video && key == "r_frame_rate") {
      parameters.r_frame_rate = parse_rational(key, value);
    } else if (is_video && key == "field_order") {
      par.field_order = static_cast<AVFieldOrder>(
          parse_enum(key, value, field_order_from_name));
    } else if (is_video && key == "color_range") {
      par.color_range = static_cast<AVColorRange>(
          parse_enum(key, value, av_color_range_from_name));
    } else if (is_video && key == "color_primaries") {
      par.color_primaries = static_cast<AVColorPrimaries>(
          parse_enum(key, value, av_color_primaries_from_name));
    } else if (is_video && key == "color_transfer_characteristic") {
      par.color_trc = static_cast<AVColorTransferCharacteristic>(
          parse_enum(key, value, av_color_transfer_from_name));
    } else if (is_video && key == "color_space") {
      par.color_space = static_cast<AVColorSpace>(
          parse_enum(key, value, av_color_space_from_name));
    } else if (is_video && key == "chroma_location") {
      par.chroma_location = static_cast<AVChromaLocation>(
          parse_enum(key, value, av_chroma_location_from_name));
    } else if (is_video && key == "video_delay") {
      par.video_delay = parse_int32(key, value);
    } else if (is_video && key == "rotation") {
      double rotation = 0;
      std::istringstream stream(value);
      stream >> rotation;
      STD_TORCH_CHECK(
          !stream.fail() && stream.eof(),
          "Invalid value for rotation: expected a number, got '",
          value,
          "'.");
      parameters.display_matrix.emplace();
      // av_display_rotation_set() takes a clockwise angle, where rotation is
      // counter-clockwise, like what av_display_rotation_get() returns.
      av_display_rotation_set(parameters.display_matrix->data(), -rotation);
    } else if (!is_video && key == "sample_format") {
      par.format = parse_enum(key, value, sample_format_from_name);
    } else if (!is_video && key == "sample_rate") {
      par.sample_rate = parse_int32(key, value);
    } else if (!is_video && key == "channel_layout") {
      set_channel_layout(par, value);
    } else if (!is_video && key == "block_align") {
      par.block_align = parse_int32(key, value);
    } else if (!is_video && key == "frame_size") {
      par.frame_size = parse_int32(key, value);
    } else if (!is_video && key == "initial_padding") {
      par.initial_padding = parse_int32(key, value);
    } else if (!is_video && key == "trailing_padding") {
      par.trailing_padding = parse_int32(key, value);
    } else if (!is_video && key == "seek_preroll") {
      par.seek_preroll = parse_int32(key, value);
    } else {
      STD_TORCH_CHECK(
          false, "Unknown ", media_type, " decoder parameter: ", key, ".");
    }
  }

  return parameters;
}

} // namespace facebook::torchcodec
