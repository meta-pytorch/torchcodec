// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "WavDecoder.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

namespace facebook::torchcodec {
namespace {

template <typename T>
T readLittleEndian(const uint8_t* data) {
  T value;
  std::memcpy(&value, data, sizeof(T));
  return value;
}

bool checkFourCC(const uint8_t* data, const char* expected) {
  return std::memcmp(data, expected, 4) == 0;
}

} // namespace

void WavDecoder::parseHeader() {
  fseek(file_, 0, SEEK_SET);

  uint8_t riffHeader[RIFF_HEADER_SIZE];
  size_t bytesRead = fread(riffHeader, 1, RIFF_HEADER_SIZE, file_);
  STD_TORCH_CHECK(
      bytesRead == RIFF_HEADER_SIZE,
      "WAV: unexpected end of data (expected ",
      RIFF_HEADER_SIZE,
      " bytes, got ",
      bytesRead,
      ")");

  STD_TORCH_CHECK(checkFourCC(riffHeader, "RIFF"), "Missing RIFF header");
  STD_TORCH_CHECK(
      checkFourCC(riffHeader + 8, "WAVE"), "Missing WAVE format identifier");

  header_.fileSize = readLittleEndian<uint32_t>(riffHeader + 4) + 8;

  uint64_t actualFileSize = std::filesystem::file_size(filePath_);
  ChunkInfo fmtChunk = findChunk("fmt ", RIFF_HEADER_SIZE, actualFileSize);
  STD_TORCH_CHECK(
      fmtChunk.size >= MIN_FMT_CHUNK_SIZE,
      "Invalid fmt chunk: size must be at least ",
      MIN_FMT_CHUNK_SIZE,
      " bytes");

  // Use ChunkInfo to seek to and read the fmt chunk data
  fseek(file_, static_cast<long>(fmtChunk.offset), SEEK_SET);
  std::vector<uint8_t> fmtData(fmtChunk.size);
  size_t fmtBytesRead = fread(fmtData.data(), 1, fmtChunk.size, file_);
  STD_TORCH_CHECK(
      fmtBytesRead == fmtChunk.size,
      "WAV: unexpected end of data (expected ",
      fmtChunk.size,
      " bytes, got ",
      fmtBytesRead,
      ")");

  header_.audioFormat = readLittleEndian<uint16_t>(fmtData.data());
  header_.numChannels = readLittleEndian<uint16_t>(fmtData.data() + 2);
  header_.sampleRate = readLittleEndian<uint32_t>(fmtData.data() + 4);
  header_.bitsPerSample = readLittleEndian<uint16_t>(fmtData.data() + 14);

  if (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) {
    STD_TORCH_CHECK(
        fmtChunk.size >= MIN_WAVEX_FMT_CHUNK_SIZE,
        "WAVE_FORMAT_EXTENSIBLE fmt chunk too small");
    header_.subFormat = readLittleEndian<uint16_t>(fmtData.data() + 24);
  }

  // TODO: Find data chunk
}

WavDecoder::WavDecoder(const std::string& path)
    : file_(nullptr), filePath_(path) {
  file_ = std::fopen(path.c_str(), "rb");
  STD_TORCH_CHECK(file_ != nullptr, "Failed to open WAV file: ", path);
  parseHeader();
  validate();
}

WavDecoder::~WavDecoder() {
  if (file_) {
    std::fclose(file_);
  }
}

uint16_t WavDecoder::getEffectiveFormat() const {
  return (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) ? header_.subFormat
                                                        : header_.audioFormat;
}

// Given a chunkId, read through each chunk until we find a match, then return
// its offset and size.
WavDecoder::ChunkInfo WavDecoder::findChunk(
    const char* chunkId,
    int64_t startPos,
    uint64_t fileSizeLimit) {
  fseek(file_, static_cast<long>(startPos), SEEK_SET);

  while (true) {
    uint8_t chunkHeader[CHUNK_HEADER_SIZE];
    size_t bytesRead = fread(chunkHeader, 1, CHUNK_HEADER_SIZE, file_);
    STD_TORCH_CHECK(
        bytesRead == CHUNK_HEADER_SIZE, "Chunk not found: ", chunkId);
    // Read chunk size which immediately follows the chunk ID
    uint32_t chunkSize = readLittleEndian<uint32_t>(chunkHeader + 4);

    if (checkFourCC(chunkHeader, chunkId)) {
      return {startPos + CHUNK_HEADER_SIZE, chunkSize};
    }

    // Skip this chunk and continue searching
    startPos += CHUNK_HEADER_SIZE + chunkSize;
    STD_TORCH_CHECK(
        static_cast<uint64_t>(startPos) <= fileSizeLimit,
        "Chunk extends beyond file bounds at position: ",
        startPos);
    fseek(file_, static_cast<long>(startPos), SEEK_SET);
  }
}

void WavDecoder::validate() const {
  uint16_t effectiveFormat = getEffectiveFormat();
  // Only uncompressed formats are supported
  if (effectiveFormat != WAV_FORMAT_PCM &&
      effectiveFormat != WAV_FORMAT_IEEE_FLOAT) {
    STD_TORCH_CHECK(
        false,
        "Unsupported WAV format: ",
        effectiveFormat,
        ". Only PCM and IEEE float formats are supported.");
  }

  if (effectiveFormat == WAV_FORMAT_PCM) {
    // TODO: support 8, 16, 24 bits
    if (header_.bitsPerSample != 32) {
      STD_TORCH_CHECK(
          false,
          "Unsupported PCM bit depth: ",
          header_.bitsPerSample,
          ". Currently supported bit depths are: 32)");
    }
  }

  // Check bit depth for IEEE_FLOAT
  if (effectiveFormat == WAV_FORMAT_IEEE_FLOAT) {
    // TODO: support 64 bit float
    if (header_.bitsPerSample != 32) {
      STD_TORCH_CHECK(
          false,
          "Unsupported IEEE_FLOAT bit depth: ",
          header_.bitsPerSample,
          ". Currently supported bit depths are: 32)");
    }
  }

  STD_TORCH_CHECK(header_.numChannels > 0, "Invalid WAV: zero channels");
  STD_TORCH_CHECK(header_.sampleRate > 0, "Invalid WAV: zero sample rate");
}

} // namespace facebook::torchcodec
