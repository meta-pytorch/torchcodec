// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include "Frame.h"
#include "Metadata.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

class FORCE_PUBLIC_VISIBILITY WavDecoder {
 public:
  explicit WavDecoder(const std::string& path);
  // Delete copy constructor and copy assignment operator since std::ifstream
  // is stored as a member variable and is not copyable.
  WavDecoder(const WavDecoder&) = delete;
  WavDecoder& operator=(const WavDecoder&) = delete;
  WavDecoder(WavDecoder&&) noexcept = default;
  WavDecoder& operator=(WavDecoder&&) noexcept = default;
  ~WavDecoder() = default;

  AudioFramesOutput getSamplesInRange(
      double startSeconds,
      std::optional<double> stopSecondsOptional = std::nullopt);

  StreamMetadata getStreamMetadata() const;

 private:
  struct WavHeader {
    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
    uint16_t numBytesPerSample =
        0; // Bytes per sample across all channels (WAV blockAlign field)
    uint16_t bitsPerSample = 0;
    uint64_t dataOffset = 0;
    // Extended format fields (WAVE_FORMAT_EXTENSIBLE)
    uint16_t subFormat = 0; // Extracted from SubFormat GUID (first 2 bytes)
    uint32_t dataSize = 0; // Size of audio data in bytes
  };

  struct ChunkInfo {
    uint64_t offset;
    uint32_t size;

    ChunkInfo(uint64_t offset, uint32_t size) : offset(offset), size(size) {}
  };

  ChunkInfo findChunk(
      std::string_view chunkId,
      uint64_t startPos,
      uint64_t fileSizeLimit);
  void parseHeader(uint64_t actualFileSize);
  void validateHeader();

  std::ifstream file_;
  WavHeader header_;
  std::string sampleFormat_;
  std::string codecName_;
};

} // namespace facebook::torchcodec
