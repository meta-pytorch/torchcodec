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

bool WavTensorReader::supportsDirectAccess() const {
  return true;
}

const uint8_t* WavTensorReader::getDirectDataPtr(
    int64_t position,
    int64_t size) {
  // Bounds check
  if (position < 0 || position + size > data_.numel()) {
    return nullptr;
  }
  return data_.data_ptr<uint8_t>() + position;
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

void WavDecoder::convertChunkToFloatDirect(
    const void* rawData,
    int64_t numSamples,
    float* outputPtr,
    int64_t totalSamples) {
  // Determine effective format (subFormat for extensible, audioFormat
  // otherwise)
  uint16_t effectiveFormat = header_.audioFormat;
  if (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) {
    effectiveFormat = header_.subFormat;
  }

  std::vector<float> tempBuffer(numSamples * header_.numChannels);

  if (effectiveFormat == WAV_FORMAT_IEEE_FLOAT) {
    if (header_.bitsPerSample == 32) {
      const float* src = reinterpret_cast<const float*>(rawData);
      for (int64_t i = 0; i < numSamples * header_.numChannels; ++i) {
        tempBuffer[i] = src[i];
      }
    } else if (header_.bitsPerSample == 64) {
      const double* src = reinterpret_cast<const double*>(rawData);
      for (int64_t i = 0; i < numSamples * header_.numChannels; ++i) {
        tempBuffer[i] = static_cast<float>(src[i]);
      }
    }
  } else {
    // Handle PCM formats
    switch (header_.bitsPerSample) {
      case 8: {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(rawData);
        for (int64_t i = 0; i < numSamples * header_.numChannels; ++i) {
          tempBuffer[i] = (static_cast<float>(src[i]) - 128.0f) / 128.0f;
        }
        break;
      }
      case 16: {
        const int16_t* src = reinterpret_cast<const int16_t*>(rawData);
        for (int64_t i = 0; i < numSamples * header_.numChannels; ++i) {
          tempBuffer[i] = static_cast<float>(src[i]) / 32768.0f;
        }
        break;
      }
      case 24: {
        // 24-bit handling
        const uint8_t* src = static_cast<const uint8_t*>(rawData);
        for (int64_t s = 0; s < numSamples; ++s) {
          for (int64_t c = 0; c < header_.numChannels; ++c) {
            const uint8_t* p = src + (s * header_.numChannels + c) * 3;
            int32_t sample = static_cast<int32_t>(p[0]) |
                (static_cast<int32_t>(p[1]) << 8) |
                (static_cast<int32_t>(p[2]) << 16);
            if (sample & 0x800000) {
              sample |= 0xFF000000;
            }
            tempBuffer[s * header_.numChannels + c] =
                static_cast<float>(sample) / 8388608.0f;
          }
        }
        break;
      }
      case 32: {
        const int32_t* src = reinterpret_cast<const int32_t*>(rawData);
        for (int64_t i = 0; i < numSamples * header_.numChannels; ++i) {
          tempBuffer[i] = static_cast<float>(src[i]) / 2147483648.0f;
        }
        break;
      }
    }
  }

  // Copy from interleaved tempBuffer to channel-first output layout
  // This preserves cache locality by processing one channel at a time
  for (int64_t c = 0; c < header_.numChannels; ++c) {
    for (int64_t s = 0; s < numSamples; ++s) {
      outputPtr[c * totalSamples + s] = tempBuffer[s * header_.numChannels + c];
    }
  }
}

torch::Tensor WavDecoder::convertSamplesToFloat(
    const void* rawData,
    int64_t numSamples,
    int64_t numChannels) {
  // Determine effective format (subFormat for extensible, audioFormat
  // otherwise)
  uint16_t effectiveFormat = header_.audioFormat;
  if (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) {
    effectiveFormat = header_.subFormat;
  }

  // Wrap the raw interleaved buffer as a (numSamples, numChannels) tensor view,
  // transpose to (numChannels, numSamples), then use .to(kFloat32) which
  // performs the dtype conversion and layout change in a single vectorized,
  // multi-threaded pass via ATen's optimized copy kernels.
  auto fromBlob = [&](const void* ptr, torch::Dtype dtype) {
    return torch::from_blob(
               const_cast<void*>(ptr),
               {numSamples, numChannels},
               torch::TensorOptions().dtype(dtype))
        .t();
  };

  if (effectiveFormat == WAV_FORMAT_IEEE_FLOAT) {
    if (header_.bitsPerSample == 32) {
      auto interleaved =
          fromBlob(reinterpret_cast<const float*>(rawData), torch::kFloat32);
      return interleaved; //.contiguous();
    } else if (header_.bitsPerSample == 64) {
      auto interleaved =
          fromBlob(reinterpret_cast<const double*>(rawData), torch::kFloat64);
      return interleaved.to(torch::kFloat32);
    }
  } else {
    switch (header_.bitsPerSample) {
      case 8: {
        auto interleaved =
            fromBlob(reinterpret_cast<const uint8_t*>(rawData), torch::kUInt8);
        return interleaved.to(torch::kFloat32).sub_(128.0f).div_(128.0f);
      }
      case 16: {
        auto interleaved =
            fromBlob(reinterpret_cast<const int16_t*>(rawData), torch::kInt16);
        return interleaved.to(torch::kFloat32).div_(32768.0f);
      }
      case 24: {
        // No 24-bit dtype; fall back to scalar loop.
        const uint8_t* src = static_cast<const uint8_t*>(rawData);
        torch::Tensor output =
            torch::empty({numChannels, numSamples}, torch::kFloat32);
        float* outPtr = output.data_ptr<float>();
        for (int64_t s = 0; s < numSamples; ++s) {
          for (int64_t c = 0; c < numChannels; ++c) {
            const uint8_t* p = src + (s * numChannels + c) * 3;
            int32_t sample = static_cast<int32_t>(p[0]) |
                (static_cast<int32_t>(p[1]) << 8) |
                (static_cast<int32_t>(p[2]) << 16);
            if (sample & 0x800000) {
              sample |= 0xFF000000;
            }
            outPtr[c * numSamples + s] =
                static_cast<float>(sample) / 8388608.0f;
          }
        }
        return output;
      }
      case 32: {
        auto interleaved =
            fromBlob(reinterpret_cast<const int32_t*>(rawData), torch::kInt32);
        return interleaved.to(torch::kFloat32).div_(2147483648.0f);
      }
    }
  }

  // Should not be reached for valid WAV files.
  return torch::empty({numChannels, numSamples}, torch::kFloat32);
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
  int64_t dataPosition = static_cast<int64_t>(header_.dataOffset) + byteOffset;

  // Zero-copy fast path for tensor inputs
  if (reader_->supportsDirectAccess()) {
    const uint8_t* directPtr =
        reader_->getDirectDataPtr(dataPosition, bytesToRead);
    if (directPtr != nullptr) {
      // Direct conversion without intermediate copy
      torch::Tensor samples =
          convertSamplesToFloat(directPtr, numSamples, header_.numChannels);
      double ptsSeconds = static_cast<double>(startSample) / header_.sampleRate;
      return std::make_tuple(samples, ptsSeconds);
    }
  }

  // Format-specific processing strategy selection
  uint16_t effectiveFormat = header_.audioFormat;
  if (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) {
    effectiveFormat = header_.subFormat;
  }

  // F32 (32-bit IEEE float) benefits from bulk operations (no conversion
  // needed)
  if (effectiveFormat == WAV_FORMAT_IEEE_FLOAT && header_.bitsPerSample == 32) {
    // Zero-copy bulk processing: read directly into pre-allocated tensor
    reader_->seek(dataPosition);

    // Use original bulk approach with PyTorch optimized operations
    std::vector<uint8_t> rawData(bytesToRead);
    int64_t bytesRead = reader_->read(rawData.data(), bytesToRead);

    if (bytesRead <= 0) {
      // Return empty tensor for read errors
      torch::Tensor samples =
          torch::empty({header_.numChannels, 0}, torch::kFloat32);
      double ptsSeconds = static_cast<double>(startSample) / header_.sampleRate;
      return std::make_tuple(samples, ptsSeconds);
    }

    // Use bulk conversion with PyTorch optimized operations (original approach)
    int64_t actualSamples = bytesRead / header_.blockAlign;
    torch::Tensor samples = convertSamplesToFloat(
        rawData.data(), actualSamples, header_.numChannels);

    double ptsSeconds = static_cast<double>(startSample) / header_.sampleRate;
    return std::make_tuple(samples, ptsSeconds);
  }

  // Chunked processing path
  // Writes directly to pre-allocated tensor memory instead of intermediate
  // buffer
  reader_->seek(dataPosition);

  torch::Tensor samples =
      torch::empty({header_.numChannels, numSamples}, torch::kFloat32);
  float* outputPtr = samples.data_ptr<float>();

  constexpr size_t CHUNK_SIZE = 64 * 1024; // 64KB chunks
  size_t alignedChunkSize =
      (CHUNK_SIZE / header_.blockAlign) * header_.blockAlign;
  if (alignedChunkSize == 0)
    alignedChunkSize = header_.blockAlign; // Ensure at least one sample

  std::vector<uint8_t> chunkBuffer(alignedChunkSize);
  int64_t totalBytesRead = 0;
  int64_t samplesProcessed = 0;

  while (totalBytesRead < bytesToRead) {
    size_t bytesToReadThisChunk = std::min(
        static_cast<size_t>(bytesToRead - totalBytesRead), alignedChunkSize);
    int64_t bytesReadThisChunk =
        reader_->read(chunkBuffer.data(), bytesToReadThisChunk);

    if (bytesReadThisChunk <= 0) {
      break; // EOF or error
    }

    // Convert this chunk directly to the output tensor
    int64_t samplesInChunk = bytesReadThisChunk / header_.blockAlign;
    if (samplesInChunk > 0) {
      convertChunkToFloatDirect(
          chunkBuffer.data(),
          samplesInChunk,
          outputPtr + samplesProcessed,
          numSamples);
      samplesProcessed += samplesInChunk;
    }

    totalBytesRead += bytesReadThisChunk;
    if (bytesReadThisChunk < static_cast<int64_t>(bytesToReadThisChunk)) {
      break; // Partial read, likely EOF
    }
  }

  // Adjust tensor size if we read less than expected
  if (samplesProcessed < numSamples) {
    samples = samples.narrow(1, 0, samplesProcessed);
  }

  // Calculate actual PTS
  double ptsSeconds = static_cast<double>(startSample) / header_.sampleRate;

  return std::make_tuple(samples, ptsSeconds);
}

} // namespace facebook::torchcodec
