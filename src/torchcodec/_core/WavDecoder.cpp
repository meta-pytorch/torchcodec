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

void readExact(WavReader* reader, void* buffer, int64_t size) {
  int64_t bytesRead = reader->read(buffer, size);
  if (bytesRead != size) {
    throw std::runtime_error(
        "WAV: unexpected end of data (expected " + std::to_string(size) +
        " bytes, got " + std::to_string(bytesRead) + ")");
  }
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
  reader_->seek(0);

  // Read and verify RIFF header (12 bytes: "RIFF" + fileSize + "WAVE")
  uint8_t riffHeader[12];
  readExact(reader_.get(), riffHeader, 12);

  if (!checkFourCC(riffHeader, "RIFF")) {
    throw std::runtime_error("Missing RIFF header");
  }
  if (!checkFourCC(riffHeader + 8, "WAVE")) {
    throw std::runtime_error("Missing WAVE format identifier");
  }

  // Find and parse fmt chunk by reading chunk headers incrementally
  int64_t pos = 12;
  bool foundFmt = false;

  while (true) {
    uint8_t chunkHeader[8];
    readExact(reader_.get(), chunkHeader, 8);

    uint32_t chunkSize = readLittleEndian<uint32_t>(chunkHeader + 4);

    if (checkFourCC(chunkHeader, "fmt ")) {
      if (chunkSize < 16) {
        throw std::runtime_error("fmt chunk too small");
      }

      std::vector<uint8_t> fmtData(chunkSize);
      readExact(reader_.get(), fmtData.data(), chunkSize);

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
        if (chunkSize < 40) {
          throw std::runtime_error(
              "WAVE_FORMAT_EXTENSIBLE fmt chunk too small");
        }

        header_.validBitsPerSample =
            readLittleEndian<uint16_t>(fmtData.data() + 18);
        header_.channelMask = readLittleEndian<uint32_t>(fmtData.data() + 20);
        // SubFormat GUID starts at offset 24, first 2 bytes are the format code
        header_.subFormat = readLittleEndian<uint16_t>(fmtData.data() + 24);
      }

      foundFmt = true;
      pos += 8 + chunkSize;
      break;
    }

    // Skip unknown chunk
    pos += 8 + chunkSize;
    reader_->seek(pos);
  }

  if (!foundFmt) {
    throw std::runtime_error("fmt chunk not found");
  }

  // Find data chunk
  while (true) {
    uint8_t chunkHeader[8];
    readExact(reader_.get(), chunkHeader, 8);

    uint32_t chunkSize = readLittleEndian<uint32_t>(chunkHeader + 4);

    if (checkFourCC(chunkHeader, "data")) {
      header_.dataSize = chunkSize;
      header_.dataOffset = pos + 8;
      return;
    }

    // Skip this chunk
    pos += 8 + chunkSize;
    reader_->seek(pos);
  }
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

  // WAV stores samples interleaved: [L R L R ...]. These loops convert to
  // float and deinterleave into channel-first layout: (numChannels, numSamples)
  // in a single pass to avoid intermediate allocations.
  //
  // Example with 2 channels (L, R) and 3 samples:
  //   Input  (interleaved):   [L0 R0 L1 R1 L2 R2]
  //                            ^read:  s * numChannels + c = 0,1,2,3,4,5
  //   Output (channel-first): [L0 L1 L2 R0 R1 R2]
  //                            ^write: c * numSamples + s = 0,1,2,3,4,5
  if (effectiveFormat == WAV_FORMAT_IEEE_FLOAT) {
    if (header_.bitsPerSample == 32) {
      const float* floatSrc = reinterpret_cast<const float*>(src);
      for (int64_t s = 0; s < numSamples; ++s) {
        for (int64_t c = 0; c < numChannels; ++c) {
          outPtr[c * numSamples + s] = floatSrc[s * numChannels + c];
        }
      }
    } else if (header_.bitsPerSample == 64) {
      const double* doubleSrc = reinterpret_cast<const double*>(src);
      for (int64_t s = 0; s < numSamples; ++s) {
        for (int64_t c = 0; c < numChannels; ++c) {
          outPtr[c * numSamples + s] =
              static_cast<float>(doubleSrc[s * numChannels + c]);
        }
      }
    }
  } else {
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
