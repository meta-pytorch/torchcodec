// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "WavDecoder.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

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

// WavFileReader implementation
WavFileReader::WavFileReader(const std::string& path) : file_(nullptr) {
  file_ = std::fopen(path.c_str(), "rb");
  if (!file_) {
    throw std::runtime_error("Failed to open WAV file: " + path);
  }
}

WavFileReader::~WavFileReader() {
  if (file_) {
    std::fclose(file_);
  }
}

int64_t WavFileReader::read(void* buffer, int64_t size) {
  if (!file_) {
    return -1;
  }
  size_t bytesRead = std::fread(buffer, 1, static_cast<size_t>(size), file_);
  return static_cast<int64_t>(bytesRead);
}

int64_t WavFileReader::seek(int64_t position) {
  if (!file_) {
    return -1;
  }
  if (std::fseek(file_, static_cast<long>(position), SEEK_SET) != 0) {
    return -1;
  }
  return position;
}

// WavTensorReader implementation
WavTensorReader::WavTensorReader(const torch::Tensor& data)
    : data_(data), currentPos_(0) {
  TORCH_CHECK(data.is_contiguous(), "WAV data tensor must be contiguous");
  TORCH_CHECK(
      data.scalar_type() == torch::kUInt8, "WAV data tensor must be uint8");
}

int64_t WavTensorReader::read(void* buffer, int64_t size) {
  int64_t available = data_.numel() - currentPos_;
  int64_t toRead = std::min(size, available);
  if (toRead <= 0) {
    return 0;
  }

  const uint8_t* src = data_.data_ptr<uint8_t>() + currentPos_;
  std::memcpy(buffer, src, static_cast<size_t>(toRead));
  currentPos_ += toRead;
  return toRead;
}

int64_t WavTensorReader::seek(int64_t position) {
  if (position < 0 || position > data_.numel()) {
    return -1;
  }
  currentPos_ = position;
  return currentPos_;
}

// WavDecoder implementation
WavDecoder::WavDecoder(std::unique_ptr<WavReader> reader)
    : reader_(std::move(reader)) {
  parseHeader();
}

bool WavDecoder::isWavFile(const void* data, size_t size) {
  if (size < 12) {
    return false;
  }
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  // Check for RIFF....WAVE
  return checkFourCC(bytes, "RIFF") && checkFourCC(bytes + 8, "WAVE");
}

void WavDecoder::parseHeader() {
  // Read enough for header parsing (typical WAV headers are < 100 bytes)
  // TODO: source?
  constexpr int64_t headerBufferSize = 256;
  std::vector<uint8_t> buffer(headerBufferSize);

  reader_->seek(0);
  int64_t bytesRead = reader_->read(buffer.data(), headerBufferSize);
  if (bytesRead < 44) {
    throw std::runtime_error("WAV data too small to contain valid header");
  }

  const uint8_t* data = buffer.data();

  // Verify RIFF header
  if (!checkFourCC(data, "RIFF")) {
    throw std::runtime_error("Missing RIFF header");
  }

  // Verify WAVE format
  if (!checkFourCC(data + 8, "WAVE")) {
    throw std::runtime_error("Missing WAVE format identifier");
  }

  // Find and parse fmt chunk
  int64_t offset = 12;
  bool foundFmt = false;

  while (offset + 8 <= bytesRead) {
    if (checkFourCC(data + offset, "fmt ")) {
      uint32_t fmtSize = readLittleEndian<uint32_t>(data + offset + 4);

      if (offset + 8 + fmtSize > bytesRead) {
        throw std::runtime_error("fmt chunk extends beyond buffer");
      }

      if (fmtSize < 16) {
        throw std::runtime_error("fmt chunk too small");
      }

      const uint8_t* fmtData = data + offset + 8;
      // TODO: explain https://en.wikipedia.org/wiki/WAV#WAV_file_header
      header_.audioFormat = readLittleEndian<uint16_t>(fmtData);
      header_.numChannels = readLittleEndian<uint16_t>(fmtData + 2);
      header_.sampleRate = readLittleEndian<uint32_t>(fmtData + 4);
      header_.byteRate = readLittleEndian<uint32_t>(fmtData + 8);
      header_.blockAlign = readLittleEndian<uint16_t>(fmtData + 12);
      header_.bitsPerSample = readLittleEndian<uint16_t>(fmtData + 14);

      // Parse extended format fields for WAVE_FORMAT_EXTENSIBLE
      if (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) {
        // Extended format requires at least 40 bytes total (16 base + 2 cbSize
        // + 22 extension)
        if (fmtSize < 40) {
          throw std::runtime_error(
              "WAVE_FORMAT_EXTENSIBLE fmt chunk too small");
        }

        header_.validBitsPerSample = readLittleEndian<uint16_t>(fmtData + 18);
        header_.channelMask = readLittleEndian<uint32_t>(fmtData + 20);
        // SubFormat GUID starts at offset 24, first 2 bytes are the format code
        header_.subFormat = readLittleEndian<uint16_t>(fmtData + 24);
      }

      foundFmt = true;
      offset += 8 + fmtSize;
      break;
    }
    // Skip unknown chunks
    uint32_t chunkSize = readLittleEndian<uint32_t>(data + offset + 4);
    offset += 8 + chunkSize;
  }

  if (!foundFmt) {
    throw std::runtime_error("fmt chunk not found");
  }

  while (offset + 8 <= bytesRead) {
    if (checkFourCC(data + offset, "data")) {
      // Parse data chunk
      header_.dataSize = readLittleEndian<uint32_t>(data + offset + 4);
      header_.dataOffset = offset + 8;
      return;
    }

    // Skip this chunk
    uint32_t chunkSize = readLittleEndian<uint32_t>(data + offset + 4);
    offset += 8 + chunkSize;
  }

  throw std::runtime_error("data chunk not found");
}

bool WavDecoder::isSupported() const {
  // Determine effective format (subFormat for extensible, audioFormat
  // otherwise)
  uint16_t effectiveFormat = header_.audioFormat;
  if (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) {
    effectiveFormat = header_.subFormat;
  }

  // Support PCM and IEEE float formats
  if (effectiveFormat != WAV_FORMAT_PCM &&
      effectiveFormat != WAV_FORMAT_IEEE_FLOAT) {
    return false;
  }

  // Validate bits per sample
  if (effectiveFormat == WAV_FORMAT_PCM) {
    if (header_.bitsPerSample != 8 && header_.bitsPerSample != 16 &&
        header_.bitsPerSample != 24 && header_.bitsPerSample != 32) {
      return false;
    }
  } else if (effectiveFormat == WAV_FORMAT_IEEE_FLOAT) {
    if (header_.bitsPerSample != 32 && header_.bitsPerSample != 64) {
      return false;
    }
  }

  return header_.numChannels > 0 && header_.sampleRate > 0 &&
      header_.blockAlign > 0;
}

bool WavDecoder::isCompatible(
    std::optional<int64_t> stream_index,
    std::optional<int64_t> sample_rate,
    std::optional<int64_t> num_channels) const {
  // WAV files only have one stream at index 0
  if (stream_index.has_value() && stream_index.value() != 0) {
    return false;
  }
  // Check sample rate matches if specified (no resampling support)
  if (sample_rate.has_value() &&
      sample_rate.value() != static_cast<int64_t>(header_.sampleRate)) {
    return false;
  }
  // Check channel count matches if specified (no remixing support)
  if (num_channels.has_value() &&
      num_channels.value() != static_cast<int64_t>(header_.numChannels)) {
    return false;
  }
  return true;
}

const WavHeader& WavDecoder::getHeader() const {
  return header_;
}

double WavDecoder::getDurationSeconds() const {
  if (header_.blockAlign == 0 || header_.sampleRate == 0) {
    return 0.0;
  }
  int64_t numSamples =
      static_cast<int64_t>(header_.dataSize) / header_.blockAlign;
  return static_cast<double>(numSamples) / header_.sampleRate;
}

torch::Tensor WavDecoder::convertSamplesToFloat(
    const void* rawData,
    int64_t numSamples,
    int64_t numChannels) {
  // Output is (numChannels, numSamples) float32
  torch::Tensor output =
      torch::empty({numChannels, numSamples}, torch::kFloat32);
  float* outPtr = output.data_ptr<float>();

  const uint8_t* src = static_cast<const uint8_t*>(rawData);
  int bytesPerSample = header_.bitsPerSample / 8;

  // Determine effective format (subFormat for extensible, audioFormat
  // otherwise)
  uint16_t effectiveFormat = header_.audioFormat;
  if (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) {
    effectiveFormat = header_.subFormat;
  }

  if (effectiveFormat == WAV_FORMAT_IEEE_FLOAT) {
    if (header_.bitsPerSample == 32) {
      // 32-bit float - just copy and deinterleave
      const float* floatSrc = reinterpret_cast<const float*>(src);
      for (int64_t s = 0; s < numSamples; ++s) {
        for (int64_t c = 0; c < numChannels; ++c) {
          outPtr[c * numSamples + s] = floatSrc[s * numChannels + c];
        }
      }
    } else if (header_.bitsPerSample == 64) {
      // 64-bit float - convert to 32-bit and deinterleave
      const double* doubleSrc = reinterpret_cast<const double*>(src);
      for (int64_t s = 0; s < numSamples; ++s) {
        for (int64_t c = 0; c < numChannels; ++c) {
          outPtr[c * numSamples + s] =
              static_cast<float>(doubleSrc[s * numChannels + c]);
        }
      }
    }
  } else {
    // PCM format - convert to normalized float
    for (int64_t s = 0; s < numSamples; ++s) {
      for (int64_t c = 0; c < numChannels; ++c) {
        const uint8_t* samplePtr = src + (s * numChannels + c) * bytesPerSample;
        float value = 0.0f;

        switch (header_.bitsPerSample) {
          case 8: {
            // 8-bit PCM is unsigned (0-255, center at 128)
            uint8_t sample = *samplePtr;
            value = (static_cast<float>(sample) - 128.0f) / 128.0f;
            break;
          }
          case 16: {
            // 16-bit PCM is signed
            int16_t sample = readLittleEndian<int16_t>(samplePtr);
            value = static_cast<float>(sample) / 32768.0f;
            break;
          }
          case 24: {
            // 24-bit PCM is signed, stored in 3 bytes little-endian
            int32_t sample = static_cast<int32_t>(samplePtr[0]) |
                (static_cast<int32_t>(samplePtr[1]) << 8) |
                (static_cast<int32_t>(samplePtr[2]) << 16);
            // Sign extend from 24 to 32 bits
            if (sample & 0x800000) {
              sample |= 0xFF000000;
            }
            value = static_cast<float>(sample) / 8388608.0f;
            break;
          }
          case 32: {
            // 32-bit PCM is signed
            int32_t sample = readLittleEndian<int32_t>(samplePtr);
            value = static_cast<float>(sample) / 2147483648.0f;
            break;
          }
        }
        outPtr[c * numSamples + s] = value;
      }
    }
  }

  return output;
}

std::tuple<torch::Tensor, double> WavDecoder::getSamplesInRange(
    double startSeconds,
    std::optional<double> stopSeconds) {
  TORCH_CHECK(startSeconds >= 0, "start_seconds must be non-negative");
  if (stopSeconds.has_value()) {
    TORCH_CHECK(
        stopSeconds.value() >= startSeconds,
        "stop_seconds must be >= start_seconds");
  }

  double duration = getDurationSeconds();
  if (startSeconds >= duration) {
    // Return empty tensor
    return std::make_tuple(
        torch::empty({header_.numChannels, 0}, torch::kFloat32), startSeconds);
  }

  double actualStop = stopSeconds.value_or(duration);
  actualStop = std::min(actualStop, duration);

  // Calculate sample range
  int64_t startSample = static_cast<int64_t>(startSeconds * header_.sampleRate);
  int64_t stopSample = static_cast<int64_t>(actualStop * header_.sampleRate);
  int64_t numSamples = stopSample - startSample;

  if (numSamples <= 0) {
    return std::make_tuple(
        torch::empty({header_.numChannels, 0}, torch::kFloat32), startSeconds);
  }

  // Calculate byte positions
  int64_t byteOffset = startSample * header_.blockAlign;
  int64_t bytesToRead = numSamples * header_.blockAlign;

  // Seek to position and read
  reader_->seek(static_cast<int64_t>(header_.dataOffset) + byteOffset);
  std::vector<uint8_t> rawData(bytesToRead);
  int64_t bytesRead = reader_->read(rawData.data(), bytesToRead);

  if (bytesRead < bytesToRead) {
    // Adjust numSamples if we couldn't read everything
    numSamples = bytesRead / header_.blockAlign;
    if (numSamples <= 0) {
      return std::make_tuple(
          torch::empty({header_.numChannels, 0}, torch::kFloat32),
          startSeconds);
    }
  }

  torch::Tensor samples =
      convertSamplesToFloat(rawData.data(), numSamples, header_.numChannels);

  // Calculate actual PTS
  double ptsSeconds = static_cast<double>(startSample) / header_.sampleRate;

  return std::make_tuple(samples, ptsSeconds);
}

} // namespace facebook::torchcodec
