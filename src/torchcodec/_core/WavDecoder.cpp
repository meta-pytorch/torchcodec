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

template <typename OutType, typename InType>
OutType read_value(const InType& data, int64_t offset) {
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
OutType safe_read_value(const InType& data, int64_t offset) {
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
  return read_value<OutType>(data, offset);
}

bool matches_four_cc(
    const uint8_t* data,
    int64_t data_size,
    int64_t offset,
    std::string_view expected) {
  STD_TORCH_CHECK(
      data_size >= 4 && offset <= data_size - 4,
      "Data array too small for FourCC comparison at offset ",
      offset);
  STD_TORCH_CHECK(offset >= 0);
  return std::memcmp(data + static_cast<size_t>(offset), expected.data(), 4) ==
      0;
}

void safe_read(AVIOContextHolder& avio, uint8_t* buf, int64_t bytes_to_read) {
  STD_TORCH_CHECK(bytes_to_read >= 0);
  int64_t total_read = 0;
  while (total_read < bytes_to_read) {
    int bytes_read = avio.read(
        buf + total_read,
        static_cast<int>(std::min(
            bytes_to_read - total_read, static_cast<int64_t>(INT_MAX))));
    STD_TORCH_CHECK(
        bytes_read > 0,
        "WAV: unexpected end of data (expected ",
        bytes_to_read,
        " bytes, got ",
        total_read,
        ")");
    total_read += bytes_read;
  }
  STD_TORCH_CHECK(
      total_read == bytes_to_read,
      "Read more bytes than requested: got ",
      total_read,
      ", expected ",
      bytes_to_read);
}

template <typename Container>
void safe_read(
    AVIOContextHolder& avio,
    Container& buffer,
    int64_t bytes_to_read) {
  static_assert(
      sizeof(typename Container::value_type) == 1,
      "Container value_type must be a 1-byte type");
  STD_TORCH_CHECK(
      static_cast<size_t>(bytes_to_read) <= buffer.size(),
      "Read size exceeds buffer length");
  safe_read(avio, reinterpret_cast<uint8_t*>(buffer.data()), bytes_to_read);
}

void safe_seek(AVIOContextHolder& avio, int64_t pos) {
  int64_t result = avio.seek(pos, SEEK_SET);
  STD_TORCH_CHECK(result >= 0, "Failed to seek to ", pos, " in WAV file");
}

} // namespace

WavDecoder::WavDecoder(std::unique_ptr<AVIOContextHolder> avio)
    : avio_(std::move(avio)) {
  STD_TORCH_CHECK(
      std::endian::native == std::endian::little,
      "WAV decoder requires little-endian architecture");
  STD_TORCH_CHECK(avio_ != nullptr, "AVIO context cannot be null");
  source_size_ = static_cast<uint64_t>(avio_->get_size());
  parse_header();
  validate_header();
}

void WavDecoder::parse_header() {
  safe_seek(*avio_, 0);

  std::array<uint8_t, RIFF_HEADER_SIZE> riff_header;
  safe_read(*avio_, riff_header, RIFF_HEADER_SIZE);

  STD_TORCH_CHECK(
      matches_four_cc(riff_header.data(), RIFF_HEADER_SIZE, 0, "RIFF"),
      "Missing RIFF header");
  STD_TORCH_CHECK(
      matches_four_cc(riff_header.data(), RIFF_HEADER_SIZE, 8, "WAVE"),
      "Missing WAVE format identifier");

  ChunkInfo fmt_chunk =
      find_chunk("fmt ", static_cast<uint64_t>(RIFF_HEADER_SIZE));
  STD_TORCH_CHECK(
      fmt_chunk.size >= MIN_FMT_CHUNK_SIZE,
      "Invalid fmt chunk: size must be at least ",
      MIN_FMT_CHUNK_SIZE,
      " bytes");

  // Use ChunkInfo to seek to and read the fmt chunk data
  safe_seek(*avio_, static_cast<int64_t>(fmt_chunk.offset));
  STD_TORCH_CHECK(
      fmt_chunk.size <= MAX_FMT_CHUNK_SIZE,
      "fmt chunk too large for allocation: ",
      fmt_chunk.size,
      " bytes, maximum allowed is ",
      MAX_FMT_CHUNK_SIZE,
      " bytes");
  std::vector<uint8_t> fmt_data(static_cast<size_t>(fmt_chunk.size));
  safe_read(*avio_, fmt_data, fmt_chunk.size);

  header_.audio_format = safe_read_value<uint16_t>(fmt_data, 0);
  header_.num_channels = safe_read_value<uint16_t>(fmt_data, 2);
  header_.sample_rate = safe_read_value<uint32_t>(fmt_data, 4);
  header_.num_bytes_per_sample = safe_read_value<uint16_t>(fmt_data, 12);
  header_.bits_per_sample = safe_read_value<uint16_t>(fmt_data, 14);

  if (header_.audio_format == WAV_FORMAT_EXTENSIBLE) {
    STD_TORCH_CHECK(
        fmt_chunk.size >= MIN_WAVEX_FMT_CHUNK_SIZE,
        "WAVE_FORMAT_EXTENSIBLE fmt chunk too small");
    header_.sub_format = safe_read_value<uint16_t>(fmt_data, 24);
  }

  ChunkInfo data_chunk =
      find_chunk("data", static_cast<uint64_t>(RIFF_HEADER_SIZE));
  header_.data_offset = data_chunk.offset;
  header_.data_size = data_chunk.size;
}

void WavDecoder::validate_header() {
  uint16_t effective_format = (header_.audio_format == WAV_FORMAT_EXTENSIBLE)
      ? header_.sub_format
      : header_.audio_format;
  STD_TORCH_CHECK(
      effective_format == WAV_FORMAT_PCM ||
          effective_format == WAV_FORMAT_IEEE_FLOAT,
      "Unsupported WAV format: ",
      effective_format,
      ". Only PCM and IEEE float formats are supported.");

  if (effective_format == WAV_FORMAT_PCM) {
    STD_TORCH_CHECK(
        header_.bits_per_sample == 8 || header_.bits_per_sample == 16 ||
            header_.bits_per_sample == 24 || header_.bits_per_sample == 32,
        "Unsupported PCM bit depth: ",
        header_.bits_per_sample,
        ". Currently supported bit depths are: 8, 16, 24, 32");
  } else {
    STD_TORCH_CHECK(
        header_.bits_per_sample == 32 || header_.bits_per_sample == 64,
        "Unsupported IEEE float bit depth: ",
        header_.bits_per_sample,
        ". Currently supported bit depths are: 32, 64");
  }

  STD_TORCH_CHECK(header_.num_channels > 0, "Invalid WAV: zero channels");
  STD_TORCH_CHECK(header_.sample_rate > 0, "Invalid WAV: zero sample rate");
  STD_TORCH_CHECK(
      header_.num_bytes_per_sample > 0, "Invalid WAV: zero block alignment");
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
      header_.num_bytes_per_sample ==
          header_.num_channels * (header_.bits_per_sample / 8),
      "Invalid WAV: block alignment (",
      header_.num_bytes_per_sample,
      ") does not match numChannels * bitsPerSample/8 (",
      header_.num_channels * (header_.bits_per_sample / 8),
      ")");

  if (effective_format == WAV_FORMAT_PCM && header_.bits_per_sample == 32) {
    sample_format_ = "s32";
    codec_name_ = "pcm_s32le";
  } else if (
      effective_format == WAV_FORMAT_PCM && header_.bits_per_sample == 24) {
    // FFmpeg decodes s24 into s32 samples (no native 24-bit type).
    sample_format_ = "s32";
    codec_name_ = "pcm_s24le";
  } else if (
      effective_format == WAV_FORMAT_PCM && header_.bits_per_sample == 16) {
    sample_format_ = "s16";
    codec_name_ = "pcm_s16le";
  } else if (
      effective_format == WAV_FORMAT_PCM && header_.bits_per_sample == 8) {
    sample_format_ = "u8";
    codec_name_ = "pcm_u8";
  } else if (
      effective_format == WAV_FORMAT_IEEE_FLOAT &&
      header_.bits_per_sample == 32) {
    sample_format_ = "flt";
    codec_name_ = "pcm_f32le";
  } else if (
      effective_format == WAV_FORMAT_IEEE_FLOAT &&
      header_.bits_per_sample == 64) {
    sample_format_ = "dbl";
    codec_name_ = "pcm_f64le";
  } else {
    STD_TORCH_CHECK(
        false,
        "Unsupported format after validation. "
        "This is a bug in TorchCodec, please report it.");
  }
}

// Given a chunkId, read through each chunk until we find a match, then return
// its offset and size.
WavDecoder::ChunkInfo WavDecoder::find_chunk(
    std::string_view chunk_id,
    uint64_t start_pos) {
  STD_TORCH_CHECK(
      source_size_ >= static_cast<uint64_t>(CHUNK_HEADER_SIZE),
      "File too small to contain chunk:",
      chunk_id);
  while (start_pos <= source_size_ - CHUNK_HEADER_SIZE) {
    safe_seek(*avio_, static_cast<int64_t>(start_pos));

    std::array<uint8_t, CHUNK_HEADER_SIZE> chunk_header;
    safe_read(*avio_, chunk_header, CHUNK_HEADER_SIZE);
    // Read chunk size which immediately follows the chunk ID
    uint32_t chunk_size = safe_read_value<uint32_t>(chunk_header, 4);

    if (matches_four_cc(chunk_header.data(), CHUNK_HEADER_SIZE, 0, chunk_id)) {
      return {start_pos + CHUNK_HEADER_SIZE, chunk_size};
    }
    // Skip this chunk and continue searching (odd chunks are padded)
    uint64_t num_bytes_to_skip = CHUNK_HEADER_SIZE +
        static_cast<uint64_t>(chunk_size) + (chunk_size % 2);
    STD_TORCH_CHECK(
        start_pos <= UINT64_MAX - num_bytes_to_skip,
        "File position arithmetic would overflow");
    start_pos += num_bytes_to_skip;
  }
  // If this code is reached, the required chunk was not found, so we error
  STD_TORCH_CHECK(false, "Chunk not found: ", chunk_id);
}

// Callers must ensure outputPtr has space for at least
// samplesInBuffer * numChannels floats.
void WavDecoder::convert_samples_to_float(
    const std::vector<uint8_t>& buffer_data,
    int64_t samples_in_buffer,
    float* output_ptr) const {
  int64_t total_samples = samples_in_buffer * header_.num_channels;

  // Normalize PCM samples to [-1.0, 1.0] range. The convention across
  // implementations is to divide by 2^(N - 1) where N is the bitdepth.
  // We use readValue because the buffer size is already validated earlier.
  // Float32 is handled directly in getSamplesInRange (no conversion
  // needed), so it doesn't appear here.
  if (header_.bits_per_sample == 64) {
    for (int64_t i = 0; i < total_samples; ++i) {
      double sample = read_value<double>(
          buffer_data, i * static_cast<int64_t>(sizeof(double)));
      output_ptr[i] = static_cast<float>(sample);
    }
  } else if (header_.bits_per_sample == 32) {
    constexpr float scale = 1.0f / static_cast<float>(1U << 31);
    for (int64_t i = 0; i < total_samples; ++i) {
      int32_t sample = read_value<int32_t>(
          buffer_data, i * static_cast<int64_t>(sizeof(int32_t)));
      output_ptr[i] = static_cast<float>(sample) * scale;
    }
  } else if (header_.bits_per_sample == 24) {
    // 24-bit samples are 3 bytes each. We shift into the *upper* 24
    // bits of an int32 so that sign extension happens naturally, then
    // reuse the same 1/(2^31) scale as 32-bit.
    constexpr float scale = 1.0f / static_cast<float>(1U << 31);
    for (int64_t i = 0; i < total_samples; ++i) {
      int64_t offset = i * 3;
      auto b0 = static_cast<uint32_t>(buffer_data[offset]);
      auto b1 = static_cast<uint32_t>(buffer_data[offset + 1]);
      auto b2 = static_cast<uint32_t>(buffer_data[offset + 2]);
      auto sample = static_cast<int32_t>((b0 << 8) | (b1 << 16) | (b2 << 24));
      output_ptr[i] = static_cast<float>(sample) * scale;
    }
  } else if (header_.bits_per_sample == 16) {
    constexpr float scale = 1.0f / static_cast<float>(1U << 15);
    for (int64_t i = 0; i < total_samples; ++i) {
      int16_t sample = read_value<int16_t>(
          buffer_data, i * static_cast<int64_t>(sizeof(int16_t)));
      output_ptr[i] = static_cast<float>(sample) * scale;
    }
  } else {
    STD_TORCH_CHECK(
        header_.bits_per_sample == 8,
        "Unsupported bit depth in convertSamplesToFloat: ",
        header_.bits_per_sample,
        ". This is a bug in TorchCodec, please report it.");
    // 8-bit WAV is *unsigned*, so we first have to center the data (- 128)
    // before scaling it.
    constexpr float scale = 1.0f / static_cast<float>(1U << 7);
    for (int64_t i = 0; i < total_samples; ++i) {
      uint8_t sample = read_value<uint8_t>(buffer_data, i);
      output_ptr[i] = (static_cast<float>(sample) - 128.0f) * scale;
    }
  }
}

AudioFramesOutput WavDecoder::get_samples_in_range(
    double start_seconds,
    std::optional<double> stop_seconds_optional) {
  // Calculate the range of samples to decode.
  // Negative startSeconds is resolved to 0 in the Python layer.
  STD_TORCH_CHECK(
      start_seconds <= INT64_MAX / header_.sample_rate,
      "startSample calculation would overflow: startSeconds * sampleRate");
  // Sample boundary alignment: round to nearest sample to avoid partial samples
  // See corresponding logic in AudioDecoder:
  // https://github.com/meta-pytorch/torchcodec/blob/910005cf5328d9d44ff8123ad540a51db9ce15b5/src/torchcodec/decoders/_audio_decoder.py#L142
  const int64_t start_sample =
      static_cast<int64_t>(std::round(start_seconds * header_.sample_rate));

  // Cap dataSize to file size to reduce risk of large tensor allocation on
  // corrupt files with incorrect dataSize.
  int64_t end_sample = static_cast<int64_t>(
      std::min(static_cast<uint64_t>(header_.data_size), source_size_) /
      header_.num_bytes_per_sample);
  if (stop_seconds_optional.has_value()) {
    STD_TORCH_CHECK(
        start_seconds <= stop_seconds_optional.value(),
        "Start seconds (",
        start_seconds,
        ") must be less than or equal to stop seconds (",
        stop_seconds_optional.value(),
        ").");
    if (start_seconds == stop_seconds_optional.value()) {
      return AudioFramesOutput{
          torch::stable::empty({header_.num_channels, 0}, kStableFloat32),
          start_seconds};
    }
    STD_TORCH_CHECK(
        stop_seconds_optional.value() <= INT64_MAX / header_.sample_rate,
        "End sample calculation would overflow: stopSeconds * sampleRate");
    int64_t requested_end_sample = static_cast<int64_t>(
        std::round(stop_seconds_optional.value() * header_.sample_rate));
    end_sample = std::min(requested_end_sample, end_sample);
  }

  const int64_t num_samples = end_sample - start_sample;
  STD_TORCH_CHECK(
      num_samples > 0,
      "No samples to decode. ",
      "This is probably because start_seconds is too high(",
      start_seconds,
      "), ",
      "or because stop_seconds is too low.");

  STD_TORCH_CHECK(
      start_sample <= INT64_MAX / header_.num_bytes_per_sample,
      "byteOffset calculation would overflow: startSample * numBytesPerSample ");
  int64_t byte_offset = start_sample * header_.num_bytes_per_sample;

  STD_TORCH_CHECK(
      header_.data_offset <= static_cast<uint64_t>(INT64_MAX - byte_offset),
      "dataPosition calculation would overflow: dataOffset + byteOffset ");
  byte_offset += static_cast<int64_t>(header_.data_offset);

  safe_seek(*avio_, byte_offset);

  auto samples =
      torch::stable::empty({num_samples, header_.num_channels}, kStableFloat32);

  if (header_.audio_format == WAV_FORMAT_IEEE_FLOAT &&
      header_.bits_per_sample == 32) {
    // Float32 samples can be read directly into the output tensor.
    int64_t total_bytes = num_samples * header_.num_bytes_per_sample;
    safe_read(
        *avio_,
        reinterpret_cast<uint8_t*>(samples.mutable_data_ptr<float>()),
        total_bytes);
  } else {
    // We need to align buffer size to actual boundaries of samples to
    // avoid reading partial samples. See
    // https://github.com/FFmpeg/FFmpeg/blob/0f600cbc16b7903703b47d23981b636c94a41c71/libavformat/wavdec.c#L786-L791
    size_t aligned_buffer_size = TMP_BUFFER_SIZE;
    aligned_buffer_size = (aligned_buffer_size / header_.num_bytes_per_sample) *
        header_.num_bytes_per_sample;
    STD_TORCH_CHECK(
        aligned_buffer_size > 0,
        "WAV bytes per sample (",
        header_.num_bytes_per_sample,
        ") exceeds buffer size (",
        TMP_BUFFER_SIZE,
        ")");

    std::vector<uint8_t> buffer(aligned_buffer_size);
    int64_t samples_processed = 0;
    const int64_t samples_per_buffer =
        static_cast<int64_t>(aligned_buffer_size) /
        header_.num_bytes_per_sample;

    while (samples_processed < num_samples) {
      const int64_t samples_this_iteration =
          std::min(num_samples - samples_processed, samples_per_buffer);
      safe_read(
          *avio_,
          buffer,
          samples_this_iteration * header_.num_bytes_per_sample);

      float* output_ptr = samples.mutable_data_ptr<float>() +
          (samples_processed * header_.num_channels);
      convert_samples_to_float(buffer, samples_this_iteration, output_ptr);
      samples_processed += samples_this_iteration;
    }
  }

  // Convert to [channels, samples]
  samples = torch::stable::transpose(samples, 0, 1);

  // We return the actual sample start time
  // (rounded to nearest sample boundary to startSeconds).
  const double actual_start_seconds =
      static_cast<double>(start_sample) / header_.sample_rate;
  return AudioFramesOutput{samples, actual_start_seconds};
}

StreamMetadata WavDecoder::get_stream_metadata() const {
  StreamMetadata metadata;
  metadata.stream_index = 0; // WAV files have single audio stream
  metadata.sample_rate = static_cast<int64_t>(header_.sample_rate);
  metadata.num_channels = static_cast<int64_t>(header_.num_channels);
  metadata.sample_format = sample_format_;
  metadata.codec_name = codec_name_;

  // Calculate duration from data size
  double bit_rate = static_cast<double>(header_.sample_rate) *
      static_cast<double>(header_.num_channels) *
      static_cast<double>(header_.bits_per_sample);
  metadata.bit_rate = bit_rate;
  metadata.duration_seconds_from_header =
      static_cast<double>(header_.data_size) * 8 / bit_rate;
  metadata.begin_stream_pts_seconds_from_content = 0.0;

  return metadata;
}
} // namespace facebook::torchcodec
