// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "WavDecoder.h"

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>
#include "ValidationUtils.h"

namespace facebook::torchcodec {
namespace {

constexpr uint32_t RIFF_HEADER_SIZE = 12; // "RIFF" + fileSize + "WAVE"
constexpr uint32_t CHUNK_HEADER_SIZE = 8; // chunkID + chunkSize
// Standard WAV fmt chunk is at least 16 bytes:
// audioFormat(2) + numChannels(2) + sampleRate(4) + byteRate(4) + blockAlign(2)
// + bitsPerSample(2)
constexpr uint32_t MIN_FMT_CHUNK_SIZE = 16;
// WAVE_FORMAT_EXTENSIBLE adds to the standard WAV fmt chunk: cbSize(2) +
// wValidBitsPerSample(2) + dwChannelMask(4) + SubFormat GUID(16) = 24 more
// bytes, total 40
constexpr uint32_t MIN_WAVEX_FMT_CHUNK_SIZE = 40;
// Arbitrary max for fmt chunk allocation - set to 5x extended format size
constexpr uint32_t MAX_FMT_CHUNK_SIZE = 200;

// See standard format codes and Wav file format used in WavHeader:
// https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
constexpr uint16_t WAV_FORMAT_PCM = 1;
constexpr uint16_t WAV_FORMAT_EXTENSIBLE = 0xFFFE;

bool isLittleEndian() {
  uint32_t x = 1;
  uint8_t first_byte;
  std::memcpy(&first_byte, &x, 1);
  return first_byte == 1;
}

template <typename T, typename Container>
T readValue(const Container& data, size_t offset) {
  static_assert(std::is_trivially_copyable_v<T>);
  static_assert(
      sizeof(typename Container::value_type) == 1,
      "Container value_type must be a 1-byte type for safe byte access");
  STD_TORCH_CHECK(
      offset <= data.size() - sizeof(T),
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

template <size_t N>
bool matchesFourCC(
    const std::array<std::byte, N>& data,
    size_t offset,
    const char* expected) {
  static_assert(N >= 4);
  STD_TORCH_CHECK(
      offset <= N - 4,
      "Data array too small for FourCC comparison at offset ",
      offset);
  return std::memcmp(data.data() + offset, expected, 4) == 0;
}

template <typename Container>
void safeReadFile(std::ifstream& file, Container& buffer, size_t bytesToRead) {
  static_assert(
      sizeof(typename Container::value_type) == 1,
      "Container value_type must be a 1-byte type for safe reinterpret_cast to char*");
  STD_TORCH_CHECK(
      bytesToRead <= buffer.size(), "Read size exceeds buffer length");
  file.read(
      reinterpret_cast<char*>(buffer.data()),
      static_cast<std::streamsize>(bytesToRead));
  STD_TORCH_CHECK(
      !file.fail() &&
          file.gcount() == static_cast<std::streamsize>(bytesToRead),
      "WAV: unexpected end of data (expected ",
      bytesToRead,
      " bytes, got ",
      file.gcount(),
      ")");
}

} // namespace

WavDecoder::WavDecoder(const std::string& path)
    : file_(path, std::ios::binary) {
  // TODO WavDecoder: Support big-endian host machines
  STD_TORCH_CHECK(
      isLittleEndian(), "WAV decoder requires little-endian architecture");
  STD_TORCH_CHECK(file_.is_open(), "Failed to open WAV file: ", path);

  uint64_t fileSize;
  try {
    fileSize = std::filesystem::file_size(path);
  } catch (const std::filesystem::filesystem_error& e) {
    STD_TORCH_CHECK(
        false, "Failed to get file size for: ", path, ". Error: ", e.what());
  }
  parseHeader(fileSize);
  validateHeader();
}

void WavDecoder::parseHeader(uint64_t fileSize) {
  file_.seekg(0, std::ios::beg);

  std::array<std::byte, RIFF_HEADER_SIZE> riffHeader;
  safeReadFile(file_, riffHeader, RIFF_HEADER_SIZE);

  STD_TORCH_CHECK(matchesFourCC(riffHeader, 0, "RIFF"), "Missing RIFF header");
  STD_TORCH_CHECK(
      matchesFourCC(riffHeader, 8, "WAVE"), "Missing WAVE format identifier");

  ChunkInfo fmtChunk =
      findChunk("fmt ", static_cast<uint64_t>(RIFF_HEADER_SIZE), fileSize);
  STD_TORCH_CHECK(
      fmtChunk.size >= MIN_FMT_CHUNK_SIZE,
      "Invalid fmt chunk: size must be at least ",
      MIN_FMT_CHUNK_SIZE,
      " bytes");

  // Use ChunkInfo to seek to and read the fmt chunk data
  file_.seekg(
      validateUint64ToStreampos(fmtChunk.offset, "fmtChunk.offset"),
      std::ios::beg);
  STD_TORCH_CHECK(
      fmtChunk.size <= MAX_FMT_CHUNK_SIZE,
      "fmt chunk too large for allocation: ",
      fmtChunk.size,
      " bytes, maximum allowed is ",
      MAX_FMT_CHUNK_SIZE,
      " bytes");
  std::vector<std::byte> fmtData(fmtChunk.size);
  safeReadFile(file_, fmtData, fmtChunk.size);

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
  // TODO WavDecoder: Support WAV_FORMAT_IEEE_FLOAT 32, 64 bit
  STD_TORCH_CHECK(
      effectiveFormat == WAV_FORMAT_PCM,
      "Unsupported WAV format: ",
      effectiveFormat,
      ". Only PCM format is supported.");

  // TODO WavDecoder: support 8, 16, 24 bits
  STD_TORCH_CHECK(
      effectiveFormat != WAV_FORMAT_PCM || header_.bitsPerSample == 32,
      "Unsupported PCM bit depth: ",
      header_.bitsPerSample,
      ". Currently supported bit depths are: 32");

  STD_TORCH_CHECK(header_.numChannels > 0, "Invalid WAV: zero channels");
  STD_TORCH_CHECK(header_.sampleRate > 0, "Invalid WAV: zero sample rate");
}

// Given a chunkId, read through each chunk until we find a match, then return
// its offset and size.
WavDecoder::ChunkInfo WavDecoder::findChunk(
    const char* chunkId,
    uint64_t startPos,
    uint64_t fileSize) {
  if (fileSize < CHUNK_HEADER_SIZE) {
    STD_TORCH_CHECK(false, "File too small to contain chunk:", chunkId);
  }
  while (startPos <= fileSize - CHUNK_HEADER_SIZE) {
    file_.seekg(validateUint64ToStreampos(startPos, "startPos"), std::ios::beg);

    std::array<std::byte, CHUNK_HEADER_SIZE> chunkHeader;
    safeReadFile(file_, chunkHeader, CHUNK_HEADER_SIZE);
    // Read chunk size which immediately follows the chunk ID
    uint32_t chunkSize = readValue<uint32_t>(chunkHeader, 4);

    if (matchesFourCC(chunkHeader, 0, chunkId)) {
      STD_TORCH_CHECK(
          startPos <= UINT64_MAX - CHUNK_HEADER_SIZE,
          "File position arithmetic would overflow");
      return {startPos + CHUNK_HEADER_SIZE, chunkSize};
    }
    STD_TORCH_CHECK(
        chunkSize <= UINT64_MAX - CHUNK_HEADER_SIZE - (chunkSize % 2),
        "Chunk size would cause overflow: ",
        chunkSize);
    uint64_t chunkLen =
        CHUNK_HEADER_SIZE + static_cast<uint64_t>(chunkSize) + (chunkSize % 2);
    STD_TORCH_CHECK(
        startPos <= UINT64_MAX - chunkLen,
        "File position arithmetic would overflow");
    // Skip this chunk and continue searching (odd chunks are padded)
    startPos += chunkLen;
  }
  STD_TORCH_CHECK(false, "Chunk not found: ", chunkId);
}

} // namespace facebook::torchcodec
