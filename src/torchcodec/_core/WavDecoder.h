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

class WavDecoder {
 public:
  explicit WavDecoder(const std::string& path);
  // Delete copy constructor and copy assignment operator since std::ifstream
  // is stored as a member variable and is not copyable.
  WavDecoder(const WavDecoder&) = delete;
  WavDecoder& operator=(const WavDecoder&) = delete;
  WavDecoder(WavDecoder&&) = default;
  WavDecoder& operator=(WavDecoder&&) = default;

 private:
  struct WavHeader {
    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
    uint16_t bitsPerSample = 0;
    // Extended format fields (WAVE_FORMAT_EXTENSIBLE)
    uint16_t subFormat = 0; // Extracted from SubFormat GUID (first 2 bytes)
  };

  struct ChunkInfo {
    uint64_t offset;
    uint32_t size;
  };

  ChunkInfo
  findChunk(const char* chunkId, uint64_t startPos, uint64_t fileSizeLimit);
  void parseHeader(uint64_t actualFileSize);
  void validateHeader() const;

  std::ifstream file_;
  WavHeader header_;
};

} // namespace facebook::torchcodec
