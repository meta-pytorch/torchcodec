// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include "StableABICompat.h"

namespace facebook::torchcodec {
constexpr size_t RIFF_HEADER_SIZE = 12; // "RIFF" + fileSize + "WAVE"
constexpr size_t CHUNK_HEADER_SIZE = 8; // chunkID + chunkSize
constexpr size_t MIN_FMT_CHUNK_SIZE = 16;
constexpr size_t MIN_WAVEX_FMT_CHUNK_SIZE = 40;

// See standard format codes and Wav file format used in WavHeader:
// https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
constexpr uint16_t WAV_FORMAT_PCM = 1;
constexpr uint16_t WAV_FORMAT_IEEE_FLOAT = 3;
constexpr uint16_t WAV_FORMAT_EXTENSIBLE = 0xFFFE;

struct WavHeader {
  uint16_t audioFormat = 0;
  uint16_t numChannels = 0;
  uint32_t sampleRate = 0;
  uint16_t bitsPerSample = 0;
  uint64_t fileSize = 0;
  // Extended format fields (WAVE_FORMAT_EXTENSIBLE)
  uint16_t subFormat = 0; // Extracted from SubFormat GUID (first 2 bytes)
};

class WavDecoder {
 public:
  explicit WavDecoder(const std::string& path);

 private:
  struct ChunkInfo {
    int64_t offset;
    uint32_t size;
  };

  ChunkInfo
  findChunk(const char* chunkId, int64_t startPos, uint64_t fileSizeLimit);
  void parseHeader(uint64_t actualFileSize);
  void validateHeader() const;

  std::ifstream file_;
  WavHeader header_;
};

} // namespace facebook::torchcodec
