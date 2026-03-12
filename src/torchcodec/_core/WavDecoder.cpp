// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "WavDecoder.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace facebook::torchcodec {
namespace {

bool is_little_endian() {
  uint32_t x = 1;
  return *(uint8_t*)&x;
}

template <typename T>
T readValue(const uint8_t* data) {
  T value;
  std::memcpy(&value, data, sizeof(T));
  return value;
}

bool checkFourCC(const uint8_t* data, const char* expected) {
  return std::memcmp(data, expected, 4) == 0;
}

} // namespace

void WavDecoder::parseHeader(uint64_t actualFileSize) {
  file_.seekg(0, std::ios::beg);

  uint8_t riffHeader[RIFF_HEADER_SIZE];
  file_.read(reinterpret_cast<char*>(riffHeader), RIFF_HEADER_SIZE);
  STD_TORCH_CHECK(
      !file_.fail() && file_.gcount() == RIFF_HEADER_SIZE,
      "WAV: unexpected end of data (expected ",
      RIFF_HEADER_SIZE,
      " bytes, got ",
      file_.gcount(),
      ")");

  STD_TORCH_CHECK(checkFourCC(riffHeader, "RIFF"), "Missing RIFF header");
  STD_TORCH_CHECK(
      checkFourCC(riffHeader + 8, "WAVE"), "Missing WAVE format identifier");

  header_.fileSize = readValue<uint32_t>(riffHeader + 4) + 8;

  ChunkInfo fmtChunk = findChunk("fmt ", RIFF_HEADER_SIZE, actualFileSize);
  STD_TORCH_CHECK(
      fmtChunk.size >= MIN_FMT_CHUNK_SIZE,
      "Invalid fmt chunk: size must be at least ",
      MIN_FMT_CHUNK_SIZE,
      " bytes");

  // Use ChunkInfo to seek to and read the fmt chunk data
  file_.seekg(fmtChunk.offset, std::ios::beg);
  std::vector<uint8_t> fmtData(fmtChunk.size);
  file_.read(reinterpret_cast<char*>(fmtData.data()), fmtChunk.size);
  STD_TORCH_CHECK(
      !file_.fail() &&
          file_.gcount() == static_cast<std::streamsize>(fmtChunk.size),
      "WAV: unexpected end of data (expected ",
      fmtChunk.size,
      " bytes, got ",
      file_.gcount(),
      ")");

  header_.audioFormat = readValue<uint16_t>(fmtData.data());
  header_.numChannels = readValue<uint16_t>(fmtData.data() + 2);
  header_.sampleRate = readValue<uint32_t>(fmtData.data() + 4);
  header_.bitsPerSample = readValue<uint16_t>(fmtData.data() + 14);

  if (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) {
    STD_TORCH_CHECK(
        fmtChunk.size >= MIN_WAVEX_FMT_CHUNK_SIZE,
        "WAVE_FORMAT_EXTENSIBLE fmt chunk too small");
    header_.subFormat = readValue<uint16_t>(fmtData.data() + 24);
  }

  // TODO WavDecoder: Find data chunk
}

WavDecoder::WavDecoder(const std::string& path) {
  // TODO WavDecoder: Support big-endian host machines
  STD_TORCH_CHECK(
      is_little_endian(), "WAV decoder requires little-endian architecture");
  file_.open(path, std::ios::binary);
  STD_TORCH_CHECK(file_.is_open(), "Failed to open WAV file: ", path);

  uint64_t actualFileSize = std::filesystem::file_size(path);
  parseHeader(actualFileSize);
  validateHeader();
}

// Given a chunkId, read through each chunk until we find a match, then return
// its offset and size.
WavDecoder::ChunkInfo WavDecoder::findChunk(
    const char* chunkId,
    int64_t startPos,
    uint64_t fileSizeLimit) {
  while (true) {
    STD_TORCH_CHECK(
        static_cast<uint64_t>(startPos) <= fileSizeLimit,
        "Chunk extends beyond file bounds at position: ",
        startPos);
    file_.seekg(startPos, std::ios::beg);

    uint8_t chunkHeader[CHUNK_HEADER_SIZE];
    file_.read(reinterpret_cast<char*>(chunkHeader), CHUNK_HEADER_SIZE);
    STD_TORCH_CHECK(
        !file_.fail() && file_.gcount() == CHUNK_HEADER_SIZE,
        "Chunk not found: ",
        chunkId);
    // Read chunk size which immediately follows the chunk ID
    uint32_t chunkSize = readValue<uint32_t>(chunkHeader + 4);

    if (checkFourCC(chunkHeader, chunkId)) {
      return {startPos + static_cast<int64_t>(CHUNK_HEADER_SIZE), chunkSize};
    }

    // Skip this chunk and continue searching (odd chunks are padded)
    startPos += CHUNK_HEADER_SIZE + chunkSize + (chunkSize % 2);
  }
}

void WavDecoder::validateHeader() const {
  uint16_t effectiveFormat = (header_.audioFormat == WAV_FORMAT_EXTENSIBLE)
      ? header_.subFormat
      : header_.audioFormat;
  STD_TORCH_CHECK(
      effectiveFormat == WAV_FORMAT_PCM ||
          effectiveFormat == WAV_FORMAT_IEEE_FLOAT,
      "Unsupported WAV format: ",
      effectiveFormat,
      ". Only PCM and IEEE float formats are supported.");

  if (effectiveFormat == WAV_FORMAT_PCM) {
    // TODO WavDecoder: support 8, 16, 24 bits
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
    // TODO WavDecoder: support 64 bit float
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
