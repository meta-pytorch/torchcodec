// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include "StableABICompat.h"

namespace facebook::torchcodec {

constexpr uint16_t WAV_FORMAT_PCM = 1;
constexpr uint16_t WAV_FORMAT_IEEE_FLOAT = 3;
constexpr uint16_t WAV_FORMAT_EXTENSIBLE = 0xFFFE;

struct WavHeader {
  uint16_t audioFormat = 0;
  uint16_t numChannels = 0;
  uint32_t sampleRate = 0;
  uint16_t blockAlign = 0;
  uint16_t bitsPerSample = 0;
  uint32_t byteRate = 0;
  uint64_t dataOffset = 0;
  uint64_t dataSize = 0;
  uint64_t fileSize = 0;
  // Extended format fields (WAVE_FORMAT_EXTENSIBLE)
  uint32_t channelMask = 0;
  uint16_t subFormat = 0; // Extracted from SubFormat GUID (first 2 bytes)
  uint16_t validBitsPerSample = 0;
};

class WavReader {
 public:
  virtual ~WavReader() = default;
  virtual int64_t read(void* buffer, int64_t size) = 0;
  virtual int64_t seek(int64_t position) = 0;
};

class WavFileReader : public WavReader {
 public:
  explicit WavFileReader(const std::string& path);
  ~WavFileReader() override;

  int64_t read(void* buffer, int64_t size) override;
  int64_t seek(int64_t position) override;

 private:
  std::FILE* file_;
};

class WavDecoder {
 public:
  explicit WavDecoder(std::unique_ptr<WavReader> reader);
  const WavHeader& getHeader() const;
  double getDurationSeconds() const;
  std::string getCodecName() const;
  std::string getSampleFormatName() const;

 private:
  struct ChunkInfo {
    int64_t offset;
    uint32_t size;
  };

  uint16_t getEffectiveFormat() const;
  ChunkInfo findChunk(const char* chunkId, int64_t startPos = 12);
  void parseHeader();
  void validate() const;

  std::unique_ptr<WavReader> reader_;
  WavHeader header_;
};

} // namespace facebook::torchcodec
