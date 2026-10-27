// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>
#include <map>
#include <optional>
#include <string>

#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

// Everything needed to decode the packets of a stream. A demuxed stream is one
// source of these (see get_decoder_parameters()), but the packets don't have to
// come from a demuxer.
struct FORCE_PUBLIC_VISIBILITY DecoderParameters {
  DecoderParameters();

  UniqueAVCodecParameters codec_parameters;
  AVRational time_base = {0, 1};
  // The stream's AVStream::r_frame_rate. NVDEC doesn't report frame
  // durations, so they're derived from it.
  AVRational r_frame_rate = {0, 1};
  // Stamped onto every decoded frame, so downstream stages can read the
  // rotation off the frame itself instead of knowing about the stream.
  std::optional<std::array<int32_t, 9>> display_matrix;
  // The MPEG-PS demuxer doesn't return proper packets just after a seek, so the
  // first ones a decoder is fed after a reset may not be decodable. See
  // PacketDecoder::send_packet().
  bool packets_may_be_misaligned_after_reset = false;
};

// The parameters of the stream at `stream_index` in `format_context`.
FORCE_PUBLIC_VISIBILITY DecoderParameters
get_decoder_parameters(const AVFormatContext& format_context, int stream_index);

// DecoderParameters as strings, which is how they cross over to Python: keyed
// by the names of the Python fields, with FFmpeg's enums by name, rationals as
// "num/den", and bytes in hex. Each coded side data entry is keyed by
// "coded_side_data:" followed by FFmpeg's name for its type. A key is absent
// when the value is unset.
FORCE_PUBLIC_VISIBILITY std::map<std::string, std::string>
decoder_parameters_to_strings(const DecoderParameters& parameters);

} // namespace facebook::torchcodec
