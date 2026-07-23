// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecodeWebp.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !TORCHCODEC_ENABLE_WEBP

namespace facebook::torchcodec {

torch::stable::Tensor decode_webp(
    [[maybe_unused]] const torch::stable::Tensor& data,
    [[maybe_unused]] int64_t mode) {
  STD_TORCH_CHECK(
      false,
      "decode_webp: torchcodec was not compiled with libwebp support. "
      "Rebuild torchcodec in an environment where libwebp (and its development "
      "headers) are available. If you see this error in a prebuilt wheel, "
      "please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

#include "webp/decode.h"
#include "webp/demux.h"
#include "webp/types.h"

#include <torch/headeronly/core/MemoryFormat.h>

#include <algorithm>
#include <cstring>

#include "Exif.h"
#include "ImageCommon.h"

namespace facebook::torchcodec {

using namespace exif_private;

namespace {

// Find the EXIF orientation by asking libwebpdemux for the "EXIF" chunk (which
// only exists in extended VP8X files and holds raw TIFF-formatted EXIF data).
// Returns -1 (i.e. no rotation) when there's no usable EXIF chunk.
int fetch_webp_exif_orientation(const uint8_t* data, size_t size) {
  WebPData webp_data;
  WebPDataInit(&webp_data);
  webp_data.bytes = data;
  webp_data.size = size;

  WebPDemuxer* demux = WebPDemux(&webp_data);
  if (demux == nullptr) {
    return -1;
  }

  int orientation = -1;
  WebPChunkIterator chunk_iter;
  if (WebPDemuxGetChunk(demux, "EXIF", 1, &chunk_iter)) {
    const uint8_t* exif = chunk_iter.chunk.bytes;
    size_t exif_size = chunk_iter.chunk.size;
    // Some encoders prefix the payload with the "Exif\0\0" marker (like the
    // JPEG APP1 segment); skip it so we're left with the TIFF header.
    constexpr size_t exif_prefix_size = 6;
    if (exif_size >= exif_prefix_size &&
        std::memcmp(exif, "Exif\0\0", exif_prefix_size) == 0) {
      exif += exif_prefix_size;
      exif_size -= exif_prefix_size;
    }
    if (exif_size > 0) {
      orientation = fetch_exif_orientation(exif, exif_size);
    }
    WebPDemuxReleaseChunkIterator(&chunk_iter);
  }

  WebPDemuxDelete(demux);
  return orientation;
}

torch::stable::Tensor decode_animated_webp(
    const uint8_t* input_ptr,
    size_t input_size,
    int64_t mode,
    bool has_alpha) {
  bool return_rgb =
      should_return_rgb(static_cast<ImageReadMode>(mode), has_alpha);
  int num_output_channels = return_rgb ? 3 : 4;

  WebPData webp_data;
  WebPDataInit(&webp_data);
  webp_data.bytes = input_ptr;
  webp_data.size = input_size;

  WebPAnimDecoderOptions dec_options;
  STD_TORCH_CHECK(
      WebPAnimDecoderOptionsInit(&dec_options),
      "WebPAnimDecoderOptionsInit failed. This is likely a version mismatch "
      "with libwebpdemux.");
  // WebPAnimDecoder can only emit RGBA/BGRA (no 3-channel RGB), so we always
  // decode RGBA and skip the alpha below if the user wants RGB.
  dec_options.color_mode = MODE_RGBA;

  WebPAnimDecoder* dec = WebPAnimDecoderNew(&webp_data, &dec_options);
  STD_TORCH_CHECK(
      dec != nullptr,
      "WebPAnimDecoderNew failed. The file is likely corrupted or truncated.");

  WebPAnimInfo anim_info;
  if (!WebPAnimDecoderGetInfo(dec, &anim_info)) {
    WebPAnimDecoderDelete(dec);
    STD_TORCH_CHECK(false, "WebPAnimDecoderGetInfo failed.");
  }

  auto num_frames = static_cast<int64_t>(anim_info.frame_count);
  auto height = static_cast<int64_t>(anim_info.canvas_height);
  auto width = static_cast<int64_t>(anim_info.canvas_width);

  auto output = torch::stable::empty(
      {num_frames, num_output_channels, height, width},
      kStableUInt8,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      torch::headeronly::MemoryFormat::ChannelsLast);
  auto output_a = mutable_accessor<uint8_t, 4>(output);
  uint8_t* output_ptr = output.mutable_data_ptr<uint8_t>();
  const int64_t frame_num_bytes =
      static_cast<int64_t>(num_output_channels) * height * width;

  int64_t frame_index = 0;
  while (WebPAnimDecoderHasMoreFrames(dec)) {
    if (frame_index >= num_frames) {
      // Guard against the decoder yielding more frames than it reported, which
      // would write past the output tensor.
      break;
    }

    // frame_rgba is an internal buffer owned by dec: we must not free it, and
    // it's overwritten on the next GetNext call (and released by
    // WebPAnimDecoderDelete), so we copy each frame out below.
    uint8_t* frame_rgba = nullptr;
    int timestamp = 0; // in ms. unused for now (we return plain tensors).
    if (!WebPAnimDecoderGetNext(dec, &frame_rgba, &timestamp)) {
      WebPAnimDecoderDelete(dec);
      STD_TORCH_CHECK(
          false,
          "WebPAnimDecoderGetNext failed at frame ",
          frame_index,
          ". The file is likely corrupted or truncated.");
    }
    if (!return_rgb) {
      std::copy(
          frame_rgba,
          frame_rgba + frame_num_bytes,
          output_ptr + frame_index * frame_num_bytes);
    } else {
      // We're dropping the decoder's alpha, so we can't std::copy.
      for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
          const uint8_t* px = frame_rgba + (y * width + x) * 4;
          output_a[frame_index][0][y][x] = px[0];
          output_a[frame_index][1][y][x] = px[1];
          output_a[frame_index][2][y][x] = px[2];
        }
      }
    }
    ++frame_index;
  }

  WebPAnimDecoderDelete(dec);

  STD_TORCH_CHECK(
      frame_index == num_frames,
      "Decoded ",
      frame_index,
      " webp frame(s) but expected ",
      num_frames,
      ". The file is likely corrupted or truncated.");

  return output;
}

torch::stable::Tensor decode_still_webp(
    const uint8_t* input_ptr,
    size_t input_size,
    int64_t mode,
    bool has_alpha) {
  bool return_rgb =
      should_return_rgb(static_cast<ImageReadMode>(mode), has_alpha);

  auto decoding_func = return_rgb ? WebPDecodeRGB : WebPDecodeRGBA;
  int num_output_channels = return_rgb ? 3 : 4;

  int width = 0;
  int height = 0;
  auto decoded_data = decoding_func(input_ptr, input_size, &width, &height);
  STD_TORCH_CHECK(
      decoded_data != nullptr,
      "Failed to decode the WebP bitstream. "
      "The file is likely corrupted or truncated.");

  auto deleter = [decoded_data](void*) { WebPFree(decoded_data); };
  auto output = torch::stable::from_blob(
      decoded_data,
      {height, width, num_output_channels},
      {width * num_output_channels, num_output_channels, 1},
      StableDevice(kStableCPU),
      kStableUInt8,
      deleter);

  return stable_permute(output, {2, 0, 1});
}

} // namespace

torch::stable::Tensor decode_webp(
    const torch::stable::Tensor& input,
    int64_t mode) {
  validate_encoded_data(input);

  auto input_ptr = input.const_data_ptr<uint8_t>();
  auto input_size = input.numel();

  WebPBitstreamFeatures features;
  auto res = WebPGetFeatures(input_ptr, input_size, &features);
  STD_TORCH_CHECK(
      res == VP8_STATUS_OK, "WebPGetFeatures failed with error code ", res);

  auto output = features.has_animation
      ? decode_animated_webp(input_ptr, input_size, mode, features.has_alpha)
      : decode_still_webp(input_ptr, input_size, mode, features.has_alpha);

  int exif_orientation = fetch_webp_exif_orientation(input_ptr, input_size);
  return exif_orientation_transform(output, exif_orientation);
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_WEBP
