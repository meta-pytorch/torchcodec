// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace facebook::torchcodec {

// WAV format constants
constexpr uint16_t WAV_FORMAT_PCM = 1;
constexpr uint16_t WAV_FORMAT_IEEE_FLOAT = 3;
constexpr uint16_t WAV_FORMAT_EXTENSIBLE = 0xFFFE;

// Parsed WAV header information
struct WavHeader {
  uint16_t audioFormat = 0; // 1 = PCM, 3 = IEEE float
  uint16_t numChannels = 0;
  uint32_t sampleRate = 0;
  uint32_t byteRate = 0;
  uint16_t blockAlign = 0;
  uint16_t bitsPerSample = 0;
  uint64_t dataOffset = 0; // Offset to start of audio data
  uint64_t dataSize = 0; // Size of audio data in bytes

  // Extended format fields (WAVE_FORMAT_EXTENSIBLE)
  uint16_t validBitsPerSample = 0;
  uint32_t channelMask = 0;
  uint16_t subFormat = 0; // Extracted from SubFormat GUID (first 2 bytes)
};

// Abstract base class for reading WAV data from different sources
class WavReader {
 public:
  virtual ~WavReader() = default;

  // Read up to `size` bytes into `buffer`. Returns bytes actually read.
  virtual int64_t read(void* buffer, int64_t size) = 0;

  // Seek to absolute position. Returns new position or -1 on error.
  virtual int64_t seek(int64_t position) = 0;
};

// WavReader implementation for file paths
class WavFileReader : public WavReader {
 public:
  explicit WavFileReader(const std::string& path);
  ~WavFileReader() override;

  int64_t read(void* buffer, int64_t size) override;
  int64_t seek(int64_t position) override;

 private:
  std::FILE* file_;
};

// WavReader implementation for tensor/bytes data
class WavTensorReader : public WavReader {
 public:
  explicit WavTensorReader(const torch::Tensor& data);

  int64_t read(void* buffer, int64_t size) override;
  int64_t seek(int64_t position) override;

 private:
  torch::Tensor data_;
  int64_t currentPos_;
};

// Main WAV decoder class
class WavDecoder {
 public:
  explicit WavDecoder(std::unique_ptr<WavReader> reader);

  // Check if this is a supported uncompressed WAV file
  // Returns true for PCM and IEEE float formats
  bool isSupported() const;

  // Check if the requested parameters are compatible with this WAV file.
  // Returns false if resampling or channel mixing would be required.
  // WAV files only have one stream, so stream_index must be 0 or nullopt.
  bool isCompatible(
      std::optional<int64_t> stream_index,
      std::optional<int64_t> sample_rate,
      std::optional<int64_t> num_channels) const;

  // Get the parsed header
  const WavHeader& getHeader() const;

  // Get samples in a time range, returns (samples, pts_seconds)
  // samples is shape (num_channels, num_samples) float32 normalized to [-1, 1]
  std::tuple<torch::Tensor, double> getSamplesInRange(
      double startSeconds,
      std::optional<double> stopSeconds);

  // Get total duration in seconds
  double getDurationSeconds() const;

  // Static helper to check if data looks like a WAV file
  static bool isWavFile(const void* data, size_t size);

 private:
  void parseHeader();
  torch::Tensor convertSamplesToFloat(
      const void* rawData,
      int64_t numSamples,
      int64_t numChannels);

  std::unique_ptr<WavReader> reader_;
  WavHeader header_;
};

} // namespace facebook::torchcodec
