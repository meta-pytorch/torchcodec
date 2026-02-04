// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "WavDecoder.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace facebook::torchcodec {
namespace {

// PCM format codes in WAV files
constexpr uint16_t WAVE_FORMAT_PCM = 1;
constexpr uint16_t WAVE_FORMAT_IEEE_FLOAT = 3;

// Read a little-endian value from raw bytes
template <typename T>
T readLE(const uint8_t* data) {
  T value;
  std::memcpy(&value, data, sizeof(T));
  return value;
}

// Check for a 4-byte identifier (FOURCC) at a given offset
bool checkFourCC(
    const uint8_t* data,
    int64_t size,
    int64_t offset,
    const char* expected) {
  if (offset + 4 > size) {
    return false;
  }
  return std::memcmp(data + offset, expected, 4) == 0;
}

// WAV header info extracted during parsing
struct WavHeaderInfo {
  int64_t dataOffset; // Byte offset to PCM data
  int64_t dataSize; // Size of PCM data in bytes
  int64_t numSamples; // Total samples per channel
  int sampleRate;
  int numChannels;
  int bitsPerSample;
  uint16_t formatCode; // 1=PCM int, 3=IEEE float
};

// Read entire file into a vector
std::optional<std::vector<uint8_t>> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return std::nullopt;
  }

  auto size = file.tellg();
  if (size <= 0) {
    return std::nullopt;
  }

  std::vector<uint8_t> data(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
    return std::nullopt;
  }

  return data;
}

// Parse WAV header from raw bytes
std::optional<WavHeaderInfo> parseWavHeader(const uint8_t* data, int64_t size) {
  // Need at least 44 bytes for minimal WAV header
  if (size < 44) {
    return std::nullopt;
  }

  // Check RIFF/WAVE signature
  if (!checkFourCC(data, size, 0, "RIFF") ||
      !checkFourCC(data, size, 8, "WAVE")) {
    return std::nullopt;
  }

  // Parse chunks to find fmt and data
  int64_t pos = 12;
  int numChannels = 0;
  int sampleRate = 0;
  int bitsPerSample = 0;
  uint16_t formatCode = 0;
  int64_t dataOffset = 0;
  int64_t dataSize = 0;
  bool foundFmt = false;
  bool foundData = false;

  while (pos + 8 <= size) {
    uint32_t chunkSize = readLE<uint32_t>(data + pos + 4);

    if (checkFourCC(data, size, pos, "fmt ")) {
      if (pos + 8 + 16 > size) {
        return std::nullopt;
      }
      const uint8_t* fmt = data + pos + 8;
      formatCode = readLE<uint16_t>(fmt);
      numChannels = readLE<uint16_t>(fmt + 2);
      sampleRate = readLE<uint32_t>(fmt + 4);
      bitsPerSample = readLE<uint16_t>(fmt + 14);
      foundFmt = true;
    } else if (checkFourCC(data, size, pos, "data")) {
      dataOffset = pos + 8;
      dataSize = chunkSize;
      foundData = true;
      break;
    }

    pos += 8 + chunkSize + (chunkSize & 1);
  }

  if (!foundFmt || !foundData) {
    return std::nullopt;
  }

  // Validate basic parameters
  if (numChannels <= 0 || sampleRate <= 0 || bitsPerSample <= 0) {
    return std::nullopt;
  }

  // Validate format/bitsPerSample combinations
  if (formatCode == WAVE_FORMAT_PCM) {
    if (bitsPerSample != 8 && bitsPerSample != 16 && bitsPerSample != 24 &&
        bitsPerSample != 32) {
      return std::nullopt;
    }
  } else if (formatCode == WAVE_FORMAT_IEEE_FLOAT) {
    if (bitsPerSample != 32 && bitsPerSample != 64) {
      return std::nullopt;
    }
  } else {
    // Unsupported format (extensible, etc.)
    return std::nullopt;
  }

  int bytesPerSample = bitsPerSample / 8;
  int64_t bytesPerFrame = numChannels * bytesPerSample;
  int64_t numSamples = dataSize / bytesPerFrame;

  return WavHeaderInfo{
      dataOffset,
      dataSize,
      numSamples,
      sampleRate,
      numChannels,
      bitsPerSample,
      formatCode};
}

// Convert 24-bit PCM samples to float32 tensor
torch::Tensor convert24BitPcmToFloat32(
    const uint8_t* pcmData,
    int64_t numSamples,
    int numChannels) {
  auto output = torch::empty({numChannels, numSamples}, torch::kFloat32);
  float* outPtr = output.data_ptr<float>();

  constexpr float scale = 1.0f / 8388608.0f;

  for (int64_t s = 0; s < numSamples; ++s) {
    for (int c = 0; c < numChannels; ++c) {
      size_t byteIdx = (s * numChannels + c) * 3;
      int32_t sample = pcmData[byteIdx] | (pcmData[byteIdx + 1] << 8) |
          (pcmData[byteIdx + 2] << 16);
      // Sign extend from 24 to 32 bits
      if (sample & 0x800000) {
        sample |= 0xFF000000;
      }
      outPtr[c * numSamples + s] = sample * scale;
    }
  }

  return output;
}

// Convert PCM data to float32 tensor
// Returns tensor of shape (numChannels, numSamples)
torch::Tensor convertPcmToFloat32(
    const uint8_t* pcmData,
    int64_t numSamples,
    int numChannels,
    uint16_t formatCode,
    int bitsPerSample) {
  const bool isMono = (numChannels == 1);

  if (formatCode == WAVE_FORMAT_PCM) {
    switch (bitsPerSample) {
      case 8: {
        auto shape = isMono ? std::vector<int64_t>{numSamples}
                            : std::vector<int64_t>{numSamples, numChannels};
        // Interpret raw bytes as uint8
        auto uintTensor = torch::from_blob(
            const_cast<uint8_t*>(pcmData), shape, torch::kUInt8);
        // Convert to float32, then normalize from [0, 255] to [-1, 1]
        auto floatTensor =
            uintTensor.to(torch::kFloat32).sub_(128.0f).div_(128.0f);
        if (isMono) {
          return floatTensor.unsqueeze(0);
        }
        return floatTensor.t().contiguous();
      }
      case 16: {
        auto shape = isMono ? std::vector<int64_t>{numSamples}
                            : std::vector<int64_t>{numSamples, numChannels};
        // Interpret raw bytes as int16
        auto intTensor = torch::from_blob(
            const_cast<uint8_t*>(pcmData), shape, torch::kInt16);
        // Convert to float32, then normalize from [-32768, 32767] to [-1, 1]
        auto floatTensor = intTensor.to(torch::kFloat32).div_(32768.0f);
        if (isMono) {
          return floatTensor.unsqueeze(0);
        }
        return floatTensor.t().contiguous();
      }
      case 24: {
        return convert24BitPcmToFloat32(pcmData, numSamples, numChannels);
      }
      case 32: {
        auto shape = isMono ? std::vector<int64_t>{numSamples}
                            : std::vector<int64_t>{numSamples, numChannels};
        // Interpret raw bytes as int32
        auto intTensor = torch::from_blob(
            const_cast<uint8_t*>(pcmData), shape, torch::kInt32);
        // Convert to float32, then normalize from [-2^31, 2^31-1] to [-1, 1]
        auto floatTensor = intTensor.to(torch::kFloat32).div_(2147483648.0f);
        if (isMono) {
          return floatTensor.unsqueeze(0);
        }
        return floatTensor.t().contiguous();
      }
    }
  } else if (formatCode == WAVE_FORMAT_IEEE_FLOAT) {
    switch (bitsPerSample) {
      case 32: {
        auto shape = isMono ? std::vector<int64_t>{numSamples}
                            : std::vector<int64_t>{numSamples, numChannels};
        // Interpret raw bytes as float32 (already normalized by convention)
        auto floatTensor = torch::from_blob(
            const_cast<uint8_t*>(pcmData), shape, torch::kFloat32);
        if (isMono) {
          return floatTensor.clone().unsqueeze(0);
        }
        return floatTensor.t().contiguous();
      }
      case 64: {
        auto shape = isMono ? std::vector<int64_t>{numSamples}
                            : std::vector<int64_t>{numSamples, numChannels};
        // Interpret raw bytes as float64
        auto doubleTensor = torch::from_blob(
            const_cast<uint8_t*>(pcmData), shape, torch::kFloat64);
        // Convert to float32 (already normalized by convention)
        auto floatTensor = doubleTensor.to(torch::kFloat32);
        if (isMono) {
          return floatTensor.unsqueeze(0);
        }
        return floatTensor.t().contiguous();
      }
    }
  }

  TORCH_CHECK(false, "Unsupported PCM format");
}

// Validate optional parameters against WAV header
// Returns true if parameters are compatible, false otherwise
bool validateWavParams(
    const WavHeaderInfo& header,
    std::optional<int64_t> stream_index,
    std::optional<int64_t> sample_rate,
    std::optional<int64_t> num_channels) {
  // WAV files only have one stream at index 0
  if (stream_index.has_value() && stream_index.value() != 0) {
    return false;
  }
  // Check sample rate matches if specified
  if (sample_rate.has_value() && sample_rate.value() != header.sampleRate) {
    return false;
  }
  // Check channel count matches if specified
  if (num_channels.has_value() && num_channels.value() != header.numChannels) {
    return false;
  }
  return true;
}

} // namespace

std::optional<WavSamples> validateAndDecodeWavFromTensor(
    const torch::Tensor& data,
    std::optional<int64_t> stream_index,
    std::optional<int64_t> sample_rate,
    std::optional<int64_t> num_channels) {
  TORCH_CHECK(
      data.is_contiguous() && data.dtype() == torch::kUInt8,
      "Input tensor must be contiguous uint8");

  const uint8_t* ptr = data.data_ptr<uint8_t>();
  int64_t size = data.numel();

  auto header = parseWavHeader(ptr, size);
  if (!header) {
    return std::nullopt;
  }

  // Validate optional parameters
  if (!validateWavParams(*header, stream_index, sample_rate, num_channels)) {
    return std::nullopt;
  }

  // Validate data bounds
  if (header->dataOffset + header->dataSize > size) {
    return std::nullopt;
  }

  auto samples = convertPcmToFloat32(
      ptr + header->dataOffset,
      header->numSamples,
      header->numChannels,
      header->formatCode,
      header->bitsPerSample);

  double durationSeconds =
      static_cast<double>(header->numSamples) / header->sampleRate;
  return WavSamples{samples, header->sampleRate, durationSeconds};
}

std::optional<WavSamples> validateAndDecodeWavFromFile(
    const std::string& path,
    std::optional<int64_t> stream_index,
    std::optional<int64_t> sample_rate,
    std::optional<int64_t> num_channels) {
  auto fileData = readFile(path);
  if (!fileData) {
    return std::nullopt;
  }

  const uint8_t* ptr = fileData->data();
  int64_t size = static_cast<int64_t>(fileData->size());

  auto header = parseWavHeader(ptr, size);
  if (!header || header->dataOffset + header->dataSize > size) {
    return std::nullopt;
  }

  // Validate optional parameters
  if (!validateWavParams(*header, stream_index, sample_rate, num_channels)) {
    return std::nullopt;
  }

  auto samples = convertPcmToFloat32(
      ptr + header->dataOffset,
      header->numSamples,
      header->numChannels,
      header->formatCode,
      header->bitsPerSample);

  double durationSeconds =
      static_cast<double>(header->numSamples) / header->sampleRate;
  return WavSamples{samples, header->sampleRate, durationSeconds};
}

} // namespace facebook::torchcodec
