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

constexpr int64_t RIFF_HEADER_SIZE = 12; // "RIFF" + fileSize + "WAVE"
constexpr int64_t CHUNK_HEADER_SIZE = 8; // chunkID + chunkSize
// Standard WAV fmt chunk is at least 16 bytes:
// audioFormat(2) + numChannels(2) + sampleRate(4) + byteRate(4) + blockAlign(2)
// + bitsPerSample(2)
constexpr int64_t MIN_FMT_CHUNK_SIZE = 16;
// WAVE_FORMAT_EXTENSIBLE adds to the standard WAV fmt chunk: cbSize(2) +
// wValidBitsPerSample(2) + dwChannelMask(4) + SubFormat GUID(16) = 24 more
// bytes, total 40
constexpr int64_t MIN_WAVEX_FMT_CHUNK_SIZE = 40;
// Arbitrary max for fmt chunk allocation - set to 5x extended format size
constexpr int64_t MAX_FMT_CHUNK_SIZE = 200;
constexpr int64_t MAX_TENSOR_SIZE = 320'000'000; // 320 MB

// See standard format codes and Wav file format used in WavHeader:
// https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
constexpr uint16_t WAV_FORMAT_PCM = 1;
constexpr uint16_t WAV_FORMAT_EXTENSIBLE = 0xFFFE;

bool isLittleEndian() {
  int64_t x = 1;
  uint8_t firstByte;
  std::memcpy(&firstByte, &x, 1);
  return firstByte == 1;
}

template <typename T, typename Container>
T readValue(const Container& data, int64_t offset) {
  static_assert(std::is_trivially_copyable_v<T>);
  static_assert(
      sizeof(typename Container::value_type) == 1,
      "Container value_type must be a 1-byte type for safe byte access");
  // readValue is always called with positive offset, so casting to size_t is
  // safe
  STD_TORCH_CHECK(
      data.size() >= sizeof(T) &&
          static_cast<size_t>(offset) <= data.size() - sizeof(T),
      "Reading ",
      sizeof(T),
      " bytes at offset ",
      offset,
      ": exceeds buffer length ",
      data.size());
  T value;
  std::memcpy(&value, data.data() + static_cast<size_t>(offset), sizeof(T));
  return value;
}

bool matchesFourCC(
    const uint8_t* data,
    int64_t dataSize,
    int64_t offset,
    std::string_view expected) {
  STD_TORCH_CHECK(
      dataSize >= 4 && offset <= dataSize - 4,
      "Data array too small for FourCC comparison at offset ",
      offset);
  // matchesFourCC is always called with non-negative offset, so casting to
  // size_t is safe
  return std::memcmp(data + static_cast<size_t>(offset), expected.data(), 4) ==
      0;
}

template <typename Container>
void safeReadFile(std::ifstream& file, Container& buffer, int64_t bytesToRead) {
  static_assert(
      sizeof(typename Container::value_type) == 1,
      "Container value_type must be a 1-byte type for safe reinterpret_cast to char*");
  STD_TORCH_CHECK(
      static_cast<size_t>(bytesToRead) <= buffer.size(),
      "Read size exceeds buffer length");
  file.read(
      reinterpret_cast<char*>(buffer.data()),
      static_cast<std::streamsize>(bytesToRead));
  STD_TORCH_CHECK(!file.fail(), "WAV: file read error");
  STD_TORCH_CHECK(
      file.gcount() == static_cast<std::streamsize>(bytesToRead),
      "WAV: unexpected end of data (expected ",
      bytesToRead,
      " bytes, got ",
      file.gcount(),
      ")");
}

void safeSeek(
    std::ifstream& file,
    std::streampos pos,
    std::ios_base::seekdir whence = std::ios::beg) {
  file.seekg(pos, whence);
  STD_TORCH_CHECK(!file.fail(), "Failed to seek to ", pos, " in WAV file");
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
  safeSeek(file_, 0, std::ios::beg);

  std::array<uint8_t, RIFF_HEADER_SIZE> riffHeader;
  safeReadFile(file_, riffHeader, RIFF_HEADER_SIZE);

  STD_TORCH_CHECK(
      matchesFourCC(riffHeader.data(), RIFF_HEADER_SIZE, 0, "RIFF"),
      "Missing RIFF header");
  STD_TORCH_CHECK(
      matchesFourCC(riffHeader.data(), RIFF_HEADER_SIZE, 8, "WAVE"),
      "Missing WAVE format identifier");

  ChunkInfo fmtChunk =
      findChunk("fmt ", static_cast<uint64_t>(RIFF_HEADER_SIZE), fileSize);
  STD_TORCH_CHECK(
      fmtChunk.size >= MIN_FMT_CHUNK_SIZE,
      "Invalid fmt chunk: size must be at least ",
      MIN_FMT_CHUNK_SIZE,
      " bytes");

  // Use ChunkInfo to seek to and read the fmt chunk data
  safeSeek(
      file_,
      validateUint64ToStreampos(fmtChunk.offset, "fmtChunk.offset"),
      std::ios::beg);
  STD_TORCH_CHECK(
      fmtChunk.size <= MAX_FMT_CHUNK_SIZE,
      "fmt chunk too large for allocation: ",
      fmtChunk.size,
      " bytes, maximum allowed is ",
      MAX_FMT_CHUNK_SIZE,
      " bytes");
  // Cast is safe since fmtChunk.size is uint32_t, and size_t is always at least
  // 32 bits
  std::vector<uint8_t> fmtData(static_cast<size_t>(fmtChunk.size));
  safeReadFile(file_, fmtData, fmtChunk.size);

  header_.audioFormat = readValue<uint16_t>(fmtData, 0);
  header_.numChannels = readValue<uint16_t>(fmtData, 2);
  header_.sampleRate = readValue<uint32_t>(fmtData, 4);
  header_.numBytesPerSample = readValue<uint16_t>(fmtData, 12);
  header_.bitsPerSample = readValue<uint16_t>(fmtData, 14);

  if (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) {
    STD_TORCH_CHECK(
        fmtChunk.size >= MIN_WAVEX_FMT_CHUNK_SIZE,
        "WAVE_FORMAT_EXTENSIBLE fmt chunk too small");
    header_.subFormat = readValue<uint16_t>(fmtData, 24);
  }

  ChunkInfo dataChunk = findChunk("data", RIFF_HEADER_SIZE, fileSize);
  header_.dataOffset = dataChunk.offset;
  header_.dataSize = dataChunk.size;
}

void WavDecoder::validateHeader() {
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
  STD_TORCH_CHECK(
      header_.numBytesPerSample > 0, "Invalid WAV: zero block alignment");

  if (effectiveFormat == WAV_FORMAT_PCM && header_.bitsPerSample == 32) {
    sampleFormat_ = "s32";
    codecName_ = "pcm_s32le";
  } else {
    STD_TORCH_CHECK(
        false,
        "Unsupported format after validation. That's unexpected, please report this to the TorchCodec repo.");
  }
}

// Given a chunkId, read through each chunk until we find a match, then return
// its offset and size.
WavDecoder::ChunkInfo WavDecoder::findChunk(
    std::string_view chunkId,
    uint64_t startPos,
    uint64_t fileSize) {
  STD_TORCH_CHECK(
      fileSize >= static_cast<uint64_t>(CHUNK_HEADER_SIZE),
      "File too small to contain chunk:",
      chunkId);
  while (startPos <= fileSize - CHUNK_HEADER_SIZE) {
    safeSeek(
        file_, validateUint64ToStreampos(startPos, "startPos"), std::ios::beg);

    std::array<uint8_t, CHUNK_HEADER_SIZE> chunkHeader;
    safeReadFile(file_, chunkHeader, CHUNK_HEADER_SIZE);
    // Read chunk size which immediately follows the chunk ID
    uint32_t chunkSize = readValue<uint32_t>(chunkHeader, 4);

    if (matchesFourCC(chunkHeader.data(), CHUNK_HEADER_SIZE, 0, chunkId)) {
      return {startPos + CHUNK_HEADER_SIZE, chunkSize};
    }
    // Skip this chunk and continue searching (odd chunks are padded)
    uint64_t numBytesToSkip =
        CHUNK_HEADER_SIZE + static_cast<uint64_t>(chunkSize) + (chunkSize % 2);
    STD_TORCH_CHECK(
        startPos <= UINT64_MAX - numBytesToSkip,
        "File position arithmetic would overflow");
    startPos += numBytesToSkip;
  }
  // If this code is reached, the required chunk was not found, so we error
  STD_TORCH_CHECK(false, "Chunk not found: ", chunkId);
}

AudioFramesOutput WavDecoder::getSamplesInRange(
    double startSeconds,
    std::optional<double> stopSecondsOptional) {
  // Calculate the range of samples to decode
  // Sample boundary alignment: round to nearest sample to avoid partial samples
  // See corresponding logic in AudioDecoder:
  // https://github.com/meta-pytorch/torchcodec/blob/910005cf5328d9d44ff8123ad540a51db9ce15b5/src/torchcodec/decoders/_audio_decoder.py#L142
  const int64_t startSample =
      static_cast<int64_t>(std::round(startSeconds * header_.sampleRate));

  int64_t endSample =
      static_cast<int64_t>(header_.dataSize / header_.numBytesPerSample);
  if (stopSecondsOptional.has_value()) {
    STD_TORCH_CHECK(
        startSeconds <= stopSecondsOptional.value(),
        "Start seconds (",
        startSeconds,
        ") must be less than or equal to stop seconds (",
        stopSecondsOptional.value(),
        ").");
    if (startSeconds == stopSecondsOptional.value()) {
      return AudioFramesOutput{
          torch::stable::empty({header_.numChannels, 0}, kStableFloat32),
          startSeconds};
    }
    int64_t requestedEndSample = static_cast<int64_t>(
        std::round(stopSecondsOptional.value() * header_.sampleRate));
    endSample = std::min(requestedEndSample, endSample);
  }

  const int64_t numSamples = endSample - startSample;
  STD_TORCH_CHECK(
      numSamples > 0,
      "No samples to decode. ",
      "This is probably because start_seconds is too high(",
      startSeconds,
      "), ",
      "or because stop_seconds is too low.");

  STD_TORCH_CHECK(
      startSample <= INT64_MAX / header_.numBytesPerSample,
      "byteOffset calculation would overflow: startSample * numBytesPerSample ");
  const int64_t byteOffset = startSample * header_.numBytesPerSample;

  STD_TORCH_CHECK(
      header_.dataOffset <= static_cast<uint64_t>(INT64_MAX - byteOffset),
      "dataPosition calculation would overflow: dataOffset + byteOffset ");
  const int64_t dataPosition =
      static_cast<int64_t>(header_.dataOffset) + byteOffset;

  // Calculate total bytes to read and read all at once
  STD_TORCH_CHECK(
      numSamples <= INT64_MAX / header_.numBytesPerSample,
      "bytesToRead calculation overflow: numSamples * numBytesPerSample");
  const int64_t bytesToRead = numSamples * header_.numBytesPerSample;

  // TODO WavDecoder: Optimize decoding and converison by processing samples in
  // fixed size buffer. For now, we use a simpler implementation: create a
  // tensor from raw bytes and convert tensor to float32
  STD_TORCH_CHECK(
      bytesToRead <= MAX_TENSOR_SIZE,
      "Audio data too large: ",
      bytesToRead,
      " bytes, maximum allowed is ",
      MAX_TENSOR_SIZE,
      " bytes");

  safeSeek(
      file_,
      validateUint64ToStreampos(dataPosition, "dataPosition"),
      std::ios::beg);

  // Cast is safe: bytesToRead <= MAX_TENSOR_SIZE (320MB)
  // which fits in size_t which is at least 32 bits
  std::vector<uint8_t> rawData(static_cast<std::size_t>(bytesToRead));
  safeReadFile(file_, rawData, bytesToRead);

  int64_t sizes[] = {numSamples, header_.numChannels};
  int64_t strides[] = {header_.numChannels, 1};
  auto rawTensor = torch::stable::from_blob(
      rawData.data(),
      {sizes, 2},
      {strides, 2},
      StableDevice(kStableCPU),
      kStableInt32,
      nullptr);

  auto samples = torch::stable::to(rawTensor, kStableFloat32);
  auto zeros = torch::stable::new_zeros(samples, samples.sizes());
  constexpr double scale =
      1.0 / static_cast<double>(std::numeric_limits<int32_t>::max());
  // Multiplication is not in the stable ABI, we use subtract with alpha as a
  // workaround. We will remove this hack once we implement buffered reading
  // optimization.
  samples = torch::stable::subtract(
      zeros, samples, -scale); // 0 - (-scale) * samples = scale * samples

  // Convert to [channels, samples]
  samples = torch::stable::transpose(samples, 0, 1);

  // We return the actual sample start time, not the requested startSeconds.
  const double actualStartSeconds =
      static_cast<double>(startSample) / header_.sampleRate;
  return AudioFramesOutput{samples, actualStartSeconds};
}

StreamMetadata WavDecoder::getStreamMetadata() const {
  StreamMetadata metadata;
  metadata.streamIndex = 0; // WAV files have single audio stream
  metadata.sampleRate = static_cast<int64_t>(header_.sampleRate);
  metadata.numChannels = static_cast<int64_t>(header_.numChannels);
  metadata.sampleFormat = sampleFormat_;
  metadata.codecName = codecName_;

  // Calculate duration from data size
  double bitRate = static_cast<double>(header_.sampleRate) *
      static_cast<double>(header_.numChannels) *
      static_cast<double>(header_.bitsPerSample);
  metadata.bitRate = bitRate;
  metadata.durationSecondsFromHeader =
      static_cast<double>(header_.dataSize) * 8 / bitRate;
  metadata.beginStreamPtsSecondsFromContent = 0.0;

  return metadata;
}
} // namespace facebook::torchcodec
