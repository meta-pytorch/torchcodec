// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "AVIOContextHolder.h"
#include "Frame.h"
#include "Metadata.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

class FORCE_PUBLIC_VISIBILITY WavDecoder {
 public:
  explicit WavDecoder(std::unique_ptr<AVIOContextHolder> avio);
  // Delete copy constructor and copy assignment operator since
  // unique_ptr is not copyable.
  WavDecoder(const WavDecoder&) = delete;
  WavDecoder& operator=(const WavDecoder&) = delete;
  WavDecoder(WavDecoder&&) noexcept = default;
  WavDecoder& operator=(WavDecoder&&) noexcept = default;
  ~WavDecoder() = default;

  AudioFramesOutput get_samples_in_range(
      double start_seconds,
      std::optional<double> stop_seconds_optional = std::nullopt);

  StreamMetadata get_stream_metadata() const;

 private:
  struct WavHeader {
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t num_bytes_per_sample =
        0; // Bytes per sample across all channels (renamed from blockAlign)
    uint16_t bits_per_sample = 0;
    uint64_t data_offset = 0;
    // Extended format fields (WAVE_FORMAT_EXTENSIBLE)
    uint16_t sub_format = 0; // Extracted from SubFormat GUID (first 2 bytes)
    uint32_t data_size = 0; // Size of audio data in bytes
  };

  struct ChunkInfo {
    uint64_t offset;
    uint32_t size;

    ChunkInfo(uint64_t offset, uint32_t size) : offset(offset), size(size) {}
  };

  ChunkInfo find_chunk(std::string_view chunk_id, uint64_t start_pos);
  void parse_header();
  void validate_header();
  void convert_samples_to_float(
      const std::vector<uint8_t>& buffer_data,
      int64_t samples_in_buffer,
      float* output_ptr) const;

  std::unique_ptr<AVIOContextHolder> avio_;
  WavHeader header_;
  uint64_t source_size_ = 0;
  std::string sample_format_;
  std::string codec_name_;
};

} // namespace facebook::torchcodec
