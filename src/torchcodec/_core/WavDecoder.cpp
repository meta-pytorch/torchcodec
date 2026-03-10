// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "WavDecoder.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "AVIOContextHolder.h"
#include "FFMPEGCommon.h"

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

void readExact(WavReader* reader, void* buffer, int64_t size) {
  int64_t bytesRead = reader->read(buffer, size);
  STD_TORCH_CHECK(
      bytesRead == size,
      "WAV: unexpected end of data (expected ",
      size,
      " bytes, got ",
      bytesRead,
      ")");
}

} // namespace

// WavFileReader implementation
WavFileReader::WavFileReader(const std::string& path)
    : WavReader(), file_(nullptr) {
  file_ = std::fopen(path.c_str(), "rb");
  STD_TORCH_CHECK(file_ != nullptr, "Failed to open WAV file: ", path);
}

WavFileReader::~WavFileReader() {
  if (file_) {
    std::fclose(file_);
  }
}

int64_t WavFileReader::read(void* buffer, int64_t size) {
  STD_TORCH_CHECK(file_ != nullptr, "WAV file handle is null");
  size_t bytesRead = std::fread(buffer, 1, static_cast<size_t>(size), file_);
  return static_cast<int64_t>(bytesRead);
}

int64_t WavFileReader::seek(int64_t position) {
  STD_TORCH_CHECK(file_ != nullptr, "WAV file handle is null");
#ifdef _WIN32
  if (_fseeki64(file_, position, SEEK_SET) != 0) {
#else
  if (fseeko(file_, static_cast<off_t>(position), SEEK_SET) != 0) {
#endif
    return -1;
  }
  return position;
}

void WavDecoder::parseHeader() {
  reader_->seek(0);

  // Verify RIFF header (12 bytes: "RIFF" + fileSize + "WAVE")
  uint8_t riffHeader[12];
  readExact(reader_.get(), riffHeader, 12);

  STD_TORCH_CHECK(checkFourCC(riffHeader, "RIFF"), "Missing RIFF header");
  STD_TORCH_CHECK(
      checkFourCC(riffHeader + 8, "WAVE"), "Missing WAVE format identifier");

  header_.fileSize = readLittleEndian<uint32_t>(riffHeader + 4) + 8;

  // Find and parse fmt chunk
  ChunkInfo fmtChunk = findChunk("fmt ");
  STD_TORCH_CHECK(
      fmtChunk.size >= 16, "Invalid fmt chunk: size must be at least 16 bytes");

  reader_->seek(fmtChunk.offset);
  std::vector<uint8_t> fmtData(fmtChunk.size);
  readExact(reader_.get(), fmtData.data(), fmtChunk.size);

  header_.audioFormat = readLittleEndian<uint16_t>(fmtData.data());
  header_.numChannels = readLittleEndian<uint16_t>(fmtData.data() + 2);
  header_.sampleRate = readLittleEndian<uint32_t>(fmtData.data() + 4);
  header_.byteRate = readLittleEndian<uint32_t>(fmtData.data() + 8);
  header_.blockAlign = readLittleEndian<uint16_t>(fmtData.data() + 12);
  header_.bitsPerSample = readLittleEndian<uint16_t>(fmtData.data() + 14);

  // Parse extended format fields for WAVE_FORMAT_EXTENSIBLE
  if (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) {
    // Extended format requires at least 40 bytes total (16 base + 2 cbSize
    // + 22 extension)
    STD_TORCH_CHECK(
        fmtChunk.size >= 40, "WAVE_FORMAT_EXTENSIBLE fmt chunk too small");

    header_.validBitsPerSample =
        readLittleEndian<uint16_t>(fmtData.data() + 18);
    header_.channelMask = readLittleEndian<uint32_t>(fmtData.data() + 20);
    // SubFormat GUID starts at offset 24, first 2 bytes are the format code
    header_.subFormat = readLittleEndian<uint16_t>(fmtData.data() + 24);
  }

  // Find data chunk
  ChunkInfo dataChunk = findChunk("data", fmtChunk.offset + fmtChunk.size);
  header_.dataSize = dataChunk.size;
  header_.dataOffset = dataChunk.offset;
}

WavDecoder::WavDecoder(std::unique_ptr<WavReader> reader)
    : reader_(std::move(reader)) {
  parseHeader();
  validate();
}

uint16_t WavDecoder::getEffectiveFormat() const {
  return (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) ? header_.subFormat
                                                        : header_.audioFormat;
}

// Given a chunkId, read through each chunk until we find a match, then return
// its offset and size.
WavDecoder::ChunkInfo WavDecoder::findChunk(
    const char* chunkId,
    int64_t startPos) {
  reader_->seek(startPos);

  while (true) {
    uint8_t chunkHeader[8];
    int64_t bytesRead = reader_->read(chunkHeader, 8);
    STD_TORCH_CHECK(bytesRead == 8, "Chunk not found: ", chunkId);
    // Read chunk size which immediately follows the chunk ID
    uint32_t chunkSize = readLittleEndian<uint32_t>(chunkHeader + 4);

    if (checkFourCC(chunkHeader, chunkId)) {
      return {startPos + 8, chunkSize};
    }

    // Skip this chunk and continue searching
    startPos += 8 + chunkSize;
    STD_TORCH_CHECK(
        static_cast<uint64_t>(startPos) <= header_.fileSize,
        "Chunk extends beyond file bounds at position: ",
        startPos);
    reader_->seek(startPos);
  }
}

double WavDecoder::getDurationSeconds() const {
  if (header_.blockAlign == 0 || header_.sampleRate == 0) {
    return 0.0;
  }
  int64_t numSamples =
      static_cast<int64_t>(header_.dataSize) / header_.blockAlign;
  return static_cast<double>(numSamples) / header_.sampleRate;
}

void WavDecoder::validate() const {
  uint16_t effectiveFormat = getEffectiveFormat();

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
  STD_TORCH_CHECK(header_.blockAlign > 0, "Invalid WAV: zero block alignment");
}

const WavHeader& WavDecoder::getHeader() const {
  return header_;
}

std::string WavDecoder::getCodecName() const {
  uint16_t effectiveFormat = getEffectiveFormat();

  if (effectiveFormat == WAV_FORMAT_PCM) {
    switch (header_.bitsPerSample) {
      case 8:
        return "pcm_u8";
      case 16:
        return "pcm_s16le";
      case 24:
        return "pcm_s24le";
      case 32:
        return "pcm_s32le";
      default:
        STD_TORCH_CHECK(
            false, "Unsupported PCM bit depth: ", header_.bitsPerSample);
    }
  } else if (effectiveFormat == WAV_FORMAT_IEEE_FLOAT) {
    switch (header_.bitsPerSample) {
      case 32:
        return "pcm_f32le";
      case 64:
        return "pcm_f64le";
      default:
        STD_TORCH_CHECK(
            false, "Unsupported IEEE float bit depth: ", header_.bitsPerSample);
    }
  }
  STD_TORCH_CHECK(false, "Unsupported WAV format: ", effectiveFormat);
}

std::string WavDecoder::getSampleFormatName() const {
  uint16_t effectiveFormat = getEffectiveFormat();

  if (effectiveFormat == WAV_FORMAT_PCM) {
    switch (header_.bitsPerSample) {
      case 8:
        return "u8";
      case 16:
        return "s16";
      case 24:
        return "s32"; // 24-bit PCM packed in 32-bit container
      case 32:
        return "s32";
      default:
        STD_TORCH_CHECK(
            false, "Unsupported PCM bit depth: ", header_.bitsPerSample);
    }
  } else if (effectiveFormat == WAV_FORMAT_IEEE_FLOAT) {
    switch (header_.bitsPerSample) {
      case 32:
        return "flt";
      case 64:
        return "dbl";
      default:
        STD_TORCH_CHECK(
            false, "Unsupported IEEE float bit depth: ", header_.bitsPerSample);
    }
  }
  STD_TORCH_CHECK(false, "Unsupported WAV format: ", effectiveFormat);
}
} // namespace facebook::torchcodec
