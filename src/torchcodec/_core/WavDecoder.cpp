// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "WavDecoder.h"

#include <cstddef>
#include <cstring>
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
// Soundfile's default buffer size. See
// https://github.com/libsndfile/libsndfile/blob/master/src/common.h#L77
constexpr size_t TMP_BUFFER_SIZE = 8192;

// See standard format codes and Wav file format used in WavHeader:
// https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
constexpr uint16_t WAV_FORMAT_PCM = 1;
constexpr uint16_t WAV_FORMAT_IEEE_FLOAT = 3;
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
  OutType value;
  std::memcpy(
      &value, data.data() + static_cast<size_t>(offset), sizeof(OutType));
  return value;
}

template <typename OutType, typename InType>
OutType safeReadValue(const InType& data, int64_t offset) {
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
  return readValue<OutType>(data, offset);
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

void safeRead(AVIOContextHolder& avio, uint8_t* buf, int64_t bytesToRead) {
  STD_TORCH_CHECK(bytesToRead >= 0);
  int64_t totalRead = 0;
  while (totalRead < bytesToRead) {
    int bytesRead = avio.read(
        buf + totalRead,
        static_cast<int>(
            std::min(bytesToRead - totalRead, static_cast<int64_t>(INT_MAX))));
    STD_TORCH_CHECK(
        bytesRead > 0,
        "WAV: unexpected end of data (expected ",
        bytesToRead,
        " bytes, got ",
        totalRead,
        ")");
    totalRead += bytesRead;
  }
  STD_TORCH_CHECK(
      totalRead == bytesToRead,
      "Read more bytes than requested: got ",
      totalRead,
      ", expected ",
      bytesToRead);
}

template <typename Container>
void safeRead(AVIOContextHolder& avio, Container& buffer, int64_t bytesToRead) {
  static_assert(
      sizeof(typename Container::value_type) == 1,
      "Container value_type must be a 1-byte type");
  STD_TORCH_CHECK(
      static_cast<size_t>(bytesToRead) <= buffer.size(),
      "Read size exceeds buffer length");
  safeRead(avio, reinterpret_cast<uint8_t*>(buffer.data()), bytesToRead);
}

void safeSeek(AVIOContextHolder& avio, int64_t pos) {
  int64_t result = avio.seek(pos, SEEK_SET);
  STD_TORCH_CHECK(result >= 0, "Failed to seek to ", pos, " in WAV file");
}

} // namespace

WavDecoder::WavDecoder(std::unique_ptr<AVIOContextHolder> avio)
    : avio_(std::move(avio)) {
  STD_TORCH_CHECK(
      isLittleEndian(), "WAV decoder requires little-endian architecture");
  STD_TORCH_CHECK(avio_ != nullptr, "AVIO context cannot be null");
  sourceSize_ = static_cast<uint64_t>(avio_->getSize());
  parseHeader();
  validateHeader();
}

void WavDecoder::parseHeader() {
  safeSeek(*avio_, 0);

  std::array<uint8_t, RIFF_HEADER_SIZE> riffHeader;
  safeRead(*avio_, riffHeader, RIFF_HEADER_SIZE);

  STD_TORCH_CHECK(
      matchesFourCC(riffHeader.data(), RIFF_HEADER_SIZE, 0, "RIFF"),
      "Missing RIFF header");
  STD_TORCH_CHECK(
      matchesFourCC(riffHeader.data(), RIFF_HEADER_SIZE, 8, "WAVE"),
      "Missing WAVE format identifier");

  ChunkInfo fmtChunk =
      findChunk("fmt ", static_cast<uint64_t>(RIFF_HEADER_SIZE));
  STD_TORCH_CHECK(
      fmtChunk.size >= MIN_FMT_CHUNK_SIZE,
      "Invalid fmt chunk: size must be at least ",
      MIN_FMT_CHUNK_SIZE,
      " bytes");

  // Use ChunkInfo to seek to and read the fmt chunk data
  safeSeek(*avio_, static_cast<int64_t>(fmtChunk.offset));
  STD_TORCH_CHECK(
      fmtChunk.size <= MAX_FMT_CHUNK_SIZE,
      "fmt chunk too large for allocation: ",
      fmtChunk.size,
      " bytes, maximum allowed is ",
      MAX_FMT_CHUNK_SIZE,
      " bytes");
  std::vector<uint8_t> fmtData(static_cast<size_t>(fmtChunk.size));
  safeRead(*avio_, fmtData, fmtChunk.size);

  header_.audioFormat = safeReadValue<uint16_t>(fmtData, 0);
  header_.numChannels = safeReadValue<uint16_t>(fmtData, 2);
  header_.sampleRate = safeReadValue<uint32_t>(fmtData, 4);
  header_.numBytesPerSample = safeReadValue<uint16_t>(fmtData, 12);
  header_.bitsPerSample = safeReadValue<uint16_t>(fmtData, 14);

  if (header_.audioFormat == WAV_FORMAT_EXTENSIBLE) {
    STD_TORCH_CHECK(
        fmtChunk.size >= MIN_WAVEX_FMT_CHUNK_SIZE,
        "WAVE_FORMAT_EXTENSIBLE fmt chunk too small");
    header_.subFormat = safeReadValue<uint16_t>(fmtData, 24);
  }

  ChunkInfo dataChunk =
      findChunk("data", static_cast<uint64_t>(RIFF_HEADER_SIZE));
  header_.dataOffset = dataChunk.offset;
  header_.dataSize = dataChunk.size;
}

void WavDecoder::validateHeader() {
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
    STD_TORCH_CHECK(
        header_.bitsPerSample == 8 || header_.bitsPerSample == 16 ||
            header_.bitsPerSample == 24 || header_.bitsPerSample == 32,
        "Unsupported PCM bit depth: ",
        header_.bitsPerSample,
        ". Currently supported bit depths are: 8, 16, 24, 32");
  } else {
    STD_TORCH_CHECK(
        header_.bitsPerSample == 32 || header_.bitsPerSample == 64,
        "Unsupported IEEE float bit depth: ",
        header_.bitsPerSample,
        ". Currently supported bit depths are: 32, 64");
  }

  STD_TORCH_CHECK(header_.numChannels > 0, "Invalid WAV: zero channels");
  STD_TORCH_CHECK(header_.sampleRate > 0, "Invalid WAV: zero sample rate");
  STD_TORCH_CHECK(
      header_.numBytesPerSample > 0, "Invalid WAV: zero block alignment");
  // The WAV spec requires numBytesPerSample == numChannels * bitsPerSample / 8.
  // https://en.wikipedia.org/wiki/WAV#WAV_file_header
  // Our output tensor has (dataSize / numBytesPerSample) * numChannels
  // elements. By validating numBytesPerSample is consistent with numChannels,
  // total tensor size is bounded by the file:
  //   dataSize / (numChannels * bitsPerSample/8) * numChannels
  //   = dataSize / (bitsPerSample/8)
  // Without this check, a corrupt numChannels could multiply tensor size
  // independently of dataSize.
  STD_TORCH_CHECK(
      header_.numBytesPerSample ==
          header_.numChannels * (header_.bitsPerSample / 8),
      "Invalid WAV: block alignment (",
      header_.numBytesPerSample,
      ") does not match numChannels * bitsPerSample/8 (",
      header_.numChannels * (header_.bitsPerSample / 8),
      ")");

  if (effectiveFormat == WAV_FORMAT_PCM && header_.bitsPerSample == 32) {
    sampleFormat_ = "s32";
    codecName_ = "pcm_s32le";
  } else if (effectiveFormat == WAV_FORMAT_PCM && header_.bitsPerSample == 24) {
    // FFmpeg decodes s24 into s32 samples (no native 24-bit type).
    sampleFormat_ = "s32";
    codecName_ = "pcm_s24le";
  } else if (effectiveFormat == WAV_FORMAT_PCM && header_.bitsPerSample == 16) {
    sampleFormat_ = "s16";
    codecName_ = "pcm_s16le";
  } else if (effectiveFormat == WAV_FORMAT_PCM && header_.bitsPerSample == 8) {
    sampleFormat_ = "u8";
    codecName_ = "pcm_u8";
  } else if (
      effectiveFormat == WAV_FORMAT_IEEE_FLOAT && header_.bitsPerSample == 32) {
    sampleFormat_ = "flt";
    codecName_ = "pcm_f32le";
  } else if (
      effectiveFormat == WAV_FORMAT_IEEE_FLOAT && header_.bitsPerSample == 64) {
    sampleFormat_ = "dbl";
    codecName_ = "pcm_f64le";
  } else {
    STD_TORCH_CHECK(
        false,
        "Unsupported format after validation. "
        "This is a bug in TorchCodec, please report it.");
  }
}

// Given a chunkId, read through each chunk until we find a match, then return
// its offset and size.
WavDecoder::ChunkInfo WavDecoder::findChunk(
    std::string_view chunkId,
    uint64_t startPos) {
  STD_TORCH_CHECK(
      sourceSize_ >= static_cast<uint64_t>(CHUNK_HEADER_SIZE),
      "File too small to contain chunk:",
      chunkId);
  while (startPos <= sourceSize_ - CHUNK_HEADER_SIZE) {
    safeSeek(*avio_, static_cast<int64_t>(startPos));

    std::array<uint8_t, CHUNK_HEADER_SIZE> chunkHeader;
    safeRead(*avio_, chunkHeader, CHUNK_HEADER_SIZE);
    // Read chunk size which immediately follows the chunk ID
    uint32_t chunkSize = safeReadValue<uint32_t>(chunkHeader, 4);

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

// Callers must ensure outputPtr has space for at least
// samplesInBuffer * numChannels floats.
void WavDecoder::convertSamplesToFloat(
    const std::vector<uint8_t>& bufferData,
    int64_t samplesInBuffer,
    float* outputPtr) const {
  int64_t totalSamples = samplesInBuffer * header_.numChannels;

  // Normalize PCM samples to [-1.0, 1.0] range. The convention across
  // implementations is to divide by 2^(N - 1) where N is the bitdepth.
  // We use readValue because the buffer size is already validated earlier.
  // Float32 is handled directly in getSamplesInRange (no conversion
  // needed), so it doesn't appear here.
  if (header_.bitsPerSample == 64) {
    for (int64_t i = 0; i < totalSamples; ++i) {
      double sample = readValue<double>(
          bufferData, i * static_cast<int64_t>(sizeof(double)));
      outputPtr[i] = static_cast<float>(sample);
    }
  } else if (header_.bitsPerSample == 32) {
    constexpr float scale = 1.0f / static_cast<float>(1U << 31);
    for (int64_t i = 0; i < totalSamples; ++i) {
      int32_t sample = readValue<int32_t>(
          bufferData, i * static_cast<int64_t>(sizeof(int32_t)));
      outputPtr[i] = static_cast<float>(sample) * scale;
    }
  } else if (header_.bitsPerSample == 24) {
    // 24-bit samples are 3 bytes each. We shift into the *upper* 24
    // bits of an int32 so that sign extension happens naturally, then
    // reuse the same 1/(2^31) scale as 32-bit.
    constexpr float scale = 1.0f / static_cast<float>(1U << 31);
    for (int64_t i = 0; i < totalSamples; ++i) {
      int64_t offset = i * 3;
      auto b0 = static_cast<uint32_t>(bufferData[offset]);
      auto b1 = static_cast<uint32_t>(bufferData[offset + 1]);
      auto b2 = static_cast<uint32_t>(bufferData[offset + 2]);
      auto sample = static_cast<int32_t>((b0 << 8) | (b1 << 16) | (b2 << 24));
      outputPtr[i] = static_cast<float>(sample) * scale;
    }
  } else if (header_.bitsPerSample == 16) {
    constexpr float scale = 1.0f / static_cast<float>(1U << 15);
    for (int64_t i = 0; i < totalSamples; ++i) {
      int16_t sample = readValue<int16_t>(
          bufferData, i * static_cast<int64_t>(sizeof(int16_t)));
      outputPtr[i] = static_cast<float>(sample) * scale;
    }
  } else {
    STD_TORCH_CHECK(
        header_.bitsPerSample == 8,
        "Unsupported bit depth in convertSamplesToFloat: ",
        header_.bitsPerSample,
        ". This is a bug in TorchCodec, please report it.");
    // 8-bit WAV is *unsigned*, so we first have to center the data (- 128)
    // before scaling it.
    constexpr float scale = 1.0f / static_cast<float>(1U << 7);
    for (int64_t i = 0; i < totalSamples; ++i) {
      uint8_t sample = readValue<uint8_t>(bufferData, i);
      outputPtr[i] = (static_cast<float>(sample) - 128.0f) * scale;
    }
  }
}

AudioFramesOutput WavDecoder::getSamplesInRange(
    double startSeconds,
    std::optional<double> stopSecondsOptional) {
  // Calculate the range of samples to decode.
  // Negative startSeconds is resolved to 0 in the Python layer.
  STD_TORCH_CHECK(
      startSeconds <= INT64_MAX / header_.sampleRate,
      "startSample calculation would overflow: startSeconds * sampleRate");
  // Sample boundary alignment: round to nearest sample to avoid partial samples
  // See corresponding logic in AudioDecoder:
  // https://github.com/meta-pytorch/torchcodec/blob/910005cf5328d9d44ff8123ad540a51db9ce15b5/src/torchcodec/decoders/_audio_decoder.py#L142
  const int64_t startSample =
      static_cast<int64_t>(std::round(startSeconds * header_.sampleRate));

  // Cap dataSize to file size to reduce risk of large tensor allocation on
  // corrupt files with incorrect dataSize.
  int64_t endSample = static_cast<int64_t>(
      std::min(static_cast<uint64_t>(header_.dataSize), sourceSize_) /
      header_.numBytesPerSample);
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

  safeSeek(*avio_, byteOffset);

  auto samples =
      torch::stable::empty({numSamples, header_.numChannels}, kStableFloat32);

  if (header_.audioFormat == WAV_FORMAT_IEEE_FLOAT &&
      header_.bitsPerSample == 32) {
    // Float32 samples can be read directly into the output tensor.
    int64_t totalBytes = numSamples * header_.numBytesPerSample;
    safeRead(
        *avio_,
        reinterpret_cast<uint8_t*>(samples.mutable_data_ptr<float>()),
        totalBytes);
  } else {
    // We need to align buffer size to actual boundaries of samples to
    // avoid reading partial samples. See
    // https://github.com/FFmpeg/FFmpeg/blob/0f600cbc16b7903703b47d23981b636c94a41c71/libavformat/wavdec.c#L786-L791
    size_t alignedBufferSize = TMP_BUFFER_SIZE;
    alignedBufferSize = (alignedBufferSize / header_.numBytesPerSample) *
        header_.numBytesPerSample;
    STD_TORCH_CHECK(
        alignedBufferSize > 0,
        "WAV bytes per sample (",
        header_.numBytesPerSample,
        ") exceeds buffer size (",
        TMP_BUFFER_SIZE,
        ")");

    std::vector<uint8_t> buffer(alignedBufferSize);
    int64_t samplesProcessed = 0;
    const int64_t samplesPerBuffer =
        static_cast<int64_t>(alignedBufferSize) / header_.numBytesPerSample;

    while (samplesProcessed < numSamples) {
      const int64_t samplesThisIteration =
          std::min(numSamples - samplesProcessed, samplesPerBuffer);
      safeRead(
          *avio_, buffer, samplesThisIteration * header_.numBytesPerSample);

      float* outputPtr = samples.mutable_data_ptr<float>() +
          (samplesProcessed * header_.numChannels);
      convertSamplesToFloat(buffer, samplesThisIteration, outputPtr);
      samplesProcessed += samplesThisIteration;
    }
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
