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

constexpr size_t RIFF_HEADER_SIZE = 12; // "RIFF" + fileSize + "WAVE"
constexpr size_t CHUNK_HEADER_SIZE = 8; // chunkID + chunkSize
constexpr size_t MIN_FMT_CHUNK_SIZE = 16;
constexpr size_t MIN_WAVEX_FMT_CHUNK_SIZE = 40;
constexpr uint32_t MAX_CHUNK_SIZE =
    1000; // 1 KB limit to prevent excessive allocation

// See standard format codes and Wav file format used in WavHeader:
// https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
constexpr uint16_t WAV_FORMAT_PCM = 1;
constexpr uint16_t WAV_FORMAT_IEEE_FLOAT = 3;
constexpr uint16_t WAV_FORMAT_EXTENSIBLE = 0xFFFE;

bool is_little_endian() {
  uint32_t x = 1;
  uint8_t first_byte;
  std::memcpy(&first_byte, &x, 1);
  return first_byte == 1;
}

// Container template requires .data() and .size() methods.
// We currently call this function on std::array.
template <typename T, typename Container>
T readValue(const Container& data, size_t offset) {
  static_assert(std::is_trivially_copyable_v<T>);
  STD_TORCH_CHECK(
      offset + sizeof(T) <= data.size(),
      "Reading ",
      sizeof(T),
      " bytes at offset ",
      offset,
      ": exceeds buffer length ",
      data.size());
  T value;
  std::memcpy(&value, data.data() + offset, sizeof(T));
  return value;
}

// The caller should ensure that 'data' has at least 4 bytes
bool matchesFourCC(const uint8_t* data, const char* expected) {
  return std::memcmp(data, expected, 4) == 0;
}

} // namespace

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

void WavDecoder::parseHeader(uint64_t actualFileSize) {
  file_.seekg(0, std::ios::beg);

  std::array<uint8_t, RIFF_HEADER_SIZE> riffHeader;
  file_.read(reinterpret_cast<char*>(riffHeader.data()), RIFF_HEADER_SIZE);
  STD_TORCH_CHECK(
      !file_.fail() && file_.gcount() == RIFF_HEADER_SIZE,
      "WAV: unexpected end of data (expected ",
      RIFF_HEADER_SIZE,
      " bytes, got ",
      file_.gcount(),
      ")");

  STD_TORCH_CHECK(
      matchesFourCC(riffHeader.data(), "RIFF"), "Missing RIFF header");
  STD_TORCH_CHECK(
      matchesFourCC(riffHeader.data() + 8, "WAVE"),
      "Missing WAVE format identifier");

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

  header_.audioFormat = readValue<uint16_t>(fmtData, 0);
  header_.numChannels = readValue<uint16_t>(fmtData, 2);
  header_.sampleRate = readValue<uint32_t>(fmtData, 4);
  header_.bitsPerSample = readValue<uint16_t>(fmtData, 14);

  if (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) {
    STD_TORCH_CHECK(
        fmtChunk.size >= MIN_WAVEX_FMT_CHUNK_SIZE,
        "WAVE_FORMAT_EXTENSIBLE fmt chunk too small");
    header_.subFormat = readValue<uint16_t>(fmtData, 24);
  }

  // TODO WavDecoder: Find data chunk
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
          ". Currently supported bit depths are: 32");
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
          ". Currently supported bit depths are: 32");
    }
  }

  STD_TORCH_CHECK(header_.numChannels > 0, "Invalid WAV: zero channels");
  STD_TORCH_CHECK(header_.sampleRate > 0, "Invalid WAV: zero sample rate");
}

// Given a chunkId, read through each chunk until we find a match, then return
// its offset and size.
WavDecoder::ChunkInfo WavDecoder::findChunk(
    const char* chunkId,
    int64_t startPos,
    uint64_t fileSizeLimit) {
  while (startPos + CHUNK_HEADER_SIZE <= fileSizeLimit) {
    file_.seekg(startPos, std::ios::beg);

    std::array<uint8_t, CHUNK_HEADER_SIZE> chunkHeader;
    file_.read(reinterpret_cast<char*>(chunkHeader.data()), CHUNK_HEADER_SIZE);
    STD_TORCH_CHECK(
        !file_.fail() && file_.gcount() == CHUNK_HEADER_SIZE,
        "Chunk not found: ",
        chunkId);
    // Read chunk size which immediately follows the chunk ID
    uint32_t chunkSize = readValue<uint32_t>(chunkHeader, 4);

    if (matchesFourCC(chunkHeader.data(), chunkId)) {
      STD_TORCH_CHECK(
          chunkSize <= MAX_CHUNK_SIZE,
          "We tried to allocate ",
          chunkId,
          " chunk of ",
          chunkSize,
          " bytes, but maximum allowed is ",
          MAX_CHUNK_SIZE,
          " bytes.");
      return {startPos + static_cast<int64_t>(CHUNK_HEADER_SIZE), chunkSize};
    }

    // Skip this chunk and continue searching (odd chunks are padded)
    startPos += CHUNK_HEADER_SIZE + chunkSize + (chunkSize % 2);
  }
  STD_TORCH_CHECK(false, "Chunk not found: ", chunkId);
}

} // namespace facebook::torchcodec
