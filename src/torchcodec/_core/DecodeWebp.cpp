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

#if !WEBP_FOUND

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
#include "webp/types.h"

#include <cstring>

#include "Exif.h"
#include "ImageCommon.h"

namespace facebook::torchcodec {

using namespace exif_private;

namespace {

// Walk the RIFF container ourselves (no libwebpdemux dependency) to find the
// EXIF orientation. WebP files are RIFF containers:
//   "RIFF" | uint32le file_size | "WEBP" | chunks...
// where each chunk is: fourcc(4) | uint32le payload_size | payload | one pad
// byte when payload_size is odd. EXIF metadata only exists in extended (VP8X)
// files, in an "EXIF" chunk holding raw TIFF-formatted EXIF data. Returns -1
// (i.e. no rotation) when there's no usable EXIF chunk.
// Note that this is very similar to our WavDecoder RIFF parsing, we could
// consider merging both.
int fetch_webp_exif_orientation(const uint8_t* data, size_t size) {
  constexpr size_t riff_header_size = 12; // "RIFF" + size + "WEBP"
  if (size < riff_header_size || std::memcmp(data, "RIFF", 4) != 0 ||
      std::memcmp(data + 8, "WEBP", 4) != 0) {
    return -1;
  }

  size_t offset = riff_header_size;
  while (offset + 8 <= size) {
    const uint8_t* fourcc = data + offset;
    uint32_t chunk_size = static_cast<uint32_t>(data[offset + 4]) |
        (static_cast<uint32_t>(data[offset + 5]) << 8) |
        (static_cast<uint32_t>(data[offset + 6]) << 16) |
        (static_cast<uint32_t>(data[offset + 7]) << 24);
    size_t payload_offset = offset + 8;
    // payload_offset <= size is guaranteed by the loop condition, so this
    // subtraction can't underflow (and avoids overflowing payload_offset +
    // chunk_size).
    if (chunk_size > size - payload_offset) {
      break; // truncated / malformed chunk
    }

    if (std::memcmp(fourcc, "EXIF", 4) == 0) {
      const uint8_t* exif = data + payload_offset;
      size_t exif_size = chunk_size;
      // Some encoders prefix the payload with the "Exif\0\0" marker (like the
      // JPEG APP1 segment); skip it so we're left with the TIFF header.
      constexpr size_t exif_prefix_size = 6;
      if (exif_size >= exif_prefix_size &&
          std::memcmp(exif, "Exif\0\0", exif_prefix_size) == 0) {
        exif += exif_prefix_size;
        exif_size -= exif_prefix_size;
      }
      if (exif_size == 0) {
        return -1;
      } else {
        return fetch_exif_orientation(exif, exif_size);
      }
    }

    // Advance past the payload and the pad byte for odd-sized chunks.
    offset = payload_offset + chunk_size + (chunk_size & 1);
  }
  return -1;
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
  // TODO_IMAGE: support animated webp files, returning an (N, C, H, W) tensor
  // like the plan calls for. This needs the WebPAnimDecoder API (libwebpdemux).
  STD_TORCH_CHECK(
      !features.has_animation, "Animated webp files are not supported.");

  auto return_rgb =
      should_return_rgb(static_cast<ImageReadMode>(mode), features.has_alpha);

  auto decoding_func = return_rgb ? WebPDecodeRGB : WebPDecodeRGBA;
  int num_output_channels = return_rgb ? 3 : 4;

  int width = 0;
  int height = 0;
  auto decoded_data = decoding_func(input_ptr, input_size, &width, &height);
  STD_TORCH_CHECK(
      decoded_data != nullptr,
      "Failed to decode the WebP bitstream (reported dimensions ",
      features.width,
      "x",
      features.height,
      "). The file is likely corrupted or truncated.");

  auto deleter = [decoded_data](void*) { WebPFree(decoded_data); };
  auto output = torch::stable::from_blob(
      decoded_data,
      {height, width, num_output_channels},
      {width * num_output_channels, num_output_channels, 1},
      StableDevice(kStableCPU),
      kStableUInt8,
      deleter);

  int exif_orientation = fetch_webp_exif_orientation(input_ptr, input_size);

  return exif_orientation_transform(
      stable_permute(output, {2, 0, 1}), exif_orientation);
}

} // namespace facebook::torchcodec

#endif // !WEBP_FOUND
