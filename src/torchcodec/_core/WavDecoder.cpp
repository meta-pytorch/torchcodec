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
// Soundfile's default chunk size. See
// https://github.com/libsndfile/libsndfile/blob/master/src/common.h#L77
constexpr size_t DEFAULT_CHUNK_BUFFER_SIZE = 8192;

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

template <typename OutType, typename InType>
OutType readValue(const InType& data, int64_t offset) {
  static_assert(std::is_trivially_copyable_v<OutType>);
  static_assert(
      sizeof(typename InType::value_type) == 1,
      "InType value_type must be a 1-byte type for safe byte access");
  STD_TORCH_CHECK(offset >= 0);
  STD_TORCH_CHECK(
      data.size() >= sizeof(OutType) &&
          static_cast<size_t>(offset) <= data.size() - sizeof(OutType),
      "Reading ",
      sizeof(OutType),
      " bytes at offset ",
      offset,
      ": exceeds buffer length ",
      data.size());
  OutType value;
  std::memcpy(
      &value, data.data() + static_cast<size_t>(offset), sizeof(OutType));
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
  STD_TORCH_CHECK(offset >= 0);
  return std::memcmp(data + static_cast<size_t>(offset), expected.data(), 4) ==
      0;
}

template <typename Container>
void safeReadFile(std::ifstream& file, Container& buffer, int64_t bytesToRead) {
  static_assert(
      sizeof(typename Container::value_type) == 1,
      "Container value_type must be a 1-byte type for safe reinterpret_cast to char*");
  STD_TORCH_CHECK(bytesToRead >= 0);
  STD_TORCH_CHECK(
      static_cast<size_t>(bytesToRead) <= buffer.size(),
      "Read size exceeds buffer length");
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

void safeSeek(
    std::ifstream& file,
    std::streampos pos,
    std::ios_base::seekdir whence = std::ios::beg) {
  file.seekg(pos, whence);
  STD_TORCH_CHECK(!file.fail(), "Failed to seek to ", pos, " in WAV file");
}

template <typename T>
inline const T* getAlignedData(const std::vector<uint8_t>& bufferData) {
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(bufferData.data()) % alignof(T) == 0,
      "Buffer not properly aligned for direct access to ",
      typeid(T).name());

  STD_TORCH_CHECK(
      bufferData.size() % sizeof(T) == 0,
      "Buffer size (",
      bufferData.size(),
      ") is not a multiple of sizeof(",
      typeid(T).name(),
      "). Malformed data?");

  return reinterpret_cast<const T*>(bufferData.data());
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

  ChunkInfo dataChunk =
      findChunk("data", static_cast<uint64_t>(RIFF_HEADER_SIZE), fileSize);
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

void WavDecoder::convertSamplesToFloat(
    const std::vector<uint8_t>& bufferData,
    int64_t samplesInBuffer,
    float* outputPtr) const {
  int64_t totalSamples = samplesInBuffer * header_.numChannels;
  int64_t bytesPerSample = header_.bitsPerSample / 8;

  STD_TORCH_CHECK(
      bufferData.size() >= totalSamples * bytesPerSample,
      "WAV block alignment mismatch: ",
      totalSamples,
      " samples require ",
      totalSamples * bytesPerSample,
      " bytes, but only ",
      bufferData.size(),
      " bytes were read.");

  // Currently only supporting 32-bit PCM
  STD_TORCH_CHECK(
      header_.bitsPerSample == 32,
      "Unsupported PCM bit depth in conversion: ",
      header_.bitsPerSample,
      ". Currently supported: 32");

  // Normalize 32-bit PCM samples to [-1.0, 1.0] range
  constexpr float scale =
      1.0f / static_cast<float>(std::numeric_limits<int32_t>::max());
  const int32_t* intData = getAlignedData<int32_t>(bufferData);
  for (int64_t i = 0; i < totalSamples; ++i) {
    int32_t sample = intData[i];
    outputPtr[i] = static_cast<float>(sample) * scale;
  }
}

AudioFramesOutput WavDecoder::getSamplesInRange(
    double startSeconds,
    std::optional<double> stopSecondsOptional) {
  // Calculate the range of samples to decode
  STD_TORCH_CHECK(
      startSeconds <= INT64_MAX / header_.sampleRate,
      "startSample calculation would overflow: startSeconds * sampleRate");
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
    STD_TORCH_CHECK(
        stopSecondsOptional.value() <= INT64_MAX / header_.sampleRate,
        "End sample calculation would overflow: stopSeconds * sampleRate");
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
  int64_t byteOffset = startSample * header_.numBytesPerSample;

  STD_TORCH_CHECK(
      header_.dataOffset <= static_cast<uint64_t>(INT64_MAX - byteOffset),
      "dataPosition calculation would overflow: dataOffset + byteOffset ");
  byteOffset += static_cast<int64_t>(header_.dataOffset);

  safeSeek(
      file_,
      validateUint64ToStreampos(byteOffset, "byteOffset"),
      std::ios::beg);

  // We need to align buffer size to actual boundaries of samples to avoid
  // reading partial samples. See
  // https://github.com/FFmpeg/FFmpeg/blob/0f600cbc16b7903703b47d23981b636c94a41c71/libavformat/wavdec.c#L786-L791
  size_t alignedBufferSize = DEFAULT_CHUNK_BUFFER_SIZE;
  // Round down to nearest multiple of numBytesPerSample to avoid partial
  // samples
  alignedBufferSize = (alignedBufferSize / header_.numBytesPerSample) *
      header_.numBytesPerSample;
  STD_TORCH_CHECK(
      alignedBufferSize > 0,
      "WAV bytes per sample (",
      header_.numBytesPerSample,
      ") exceeds buffer size (",
      DEFAULT_CHUNK_BUFFER_SIZE,
      ")");

  // Allocate buffer and read samples in chunks
  std::vector<uint8_t> buffer(alignedBufferSize);

  int64_t totalBytesRead = 0;
  int64_t samplesProcessed = 0;
  auto samples =
      torch::stable::empty({numSamples, header_.numChannels}, kStableFloat32);

  const int64_t bytesToRead = numSamples * header_.numBytesPerSample;

  while (totalBytesRead < bytesToRead) {
    const int64_t bytesToReadThisIteration = std::min(
        bytesToRead - totalBytesRead, static_cast<int64_t>(alignedBufferSize));
    safeReadFile(file_, buffer, bytesToReadThisIteration);

    const int64_t samplesInBuffer =
        bytesToReadThisIteration / header_.numBytesPerSample;
    float* outputPtr = samples.mutable_data_ptr<float>() +
        (samplesProcessed * header_.numChannels);
    convertSamplesToFloat(buffer, samplesInBuffer, outputPtr);
    samplesProcessed += samplesInBuffer;
    totalBytesRead += bytesToReadThisIteration;
  }

  if (samplesProcessed < numSamples) {
    samples = torch::stable::narrow(samples, 0, 0, samplesProcessed);
  }
  // Convert to [channels, samples]
  samples = torch::stable::transpose(samples, 0, 1);

  // We return the actual sample start time
  // (rounded to nearest sample boundary to startSeconds).
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
