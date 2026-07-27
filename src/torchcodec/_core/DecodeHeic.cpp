// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecodeHeic.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !TORCHCODEC_ENABLE_HEIC

namespace facebook::torchcodec {

torch::stable::Tensor decode_heic(
    [[maybe_unused]] const torch::stable::Tensor& input,
    [[maybe_unused]] int64_t mode) {
  STD_TORCH_CHECK(
      false,
      "decode_heic: torchcodec was not compiled with libheif support. "
      "Rebuild torchcodec in an environment where libheif (and its development "
      "headers) are available, e.g. `conda install -c conda-forge libheif`.");
}

} // namespace facebook::torchcodec

#else

#include <bit>
#include <cstring>
#include <memory>

#include "libheif/heif.h"

#include "ImageCommon.h"

namespace facebook::torchcodec {

namespace {

struct HeifContextDeleter {
  void operator()(heif_context* ctx) const {
    heif_context_free(ctx);
  }
};

struct HeifImageHandleDeleter {
  void operator()(heif_image_handle* handle) const {
    heif_image_handle_release(handle);
  }
};

struct HeifImageDeleter {
  void operator()(heif_image* img) const {
    heif_image_release(img);
  }
};

using UniqueHeifContext = std::unique_ptr<heif_context, HeifContextDeleter>;
using UniqueHeifImageHandle =
    std::unique_ptr<heif_image_handle, HeifImageHandleDeleter>;
using UniqueHeifImage = std::unique_ptr<heif_image, HeifImageDeleter>;

} // namespace

torch::stable::Tensor decode_heic(
    const torch::stable::Tensor& input,
    int64_t mode) {
  validate_encoded_data(input);

  UniqueHeifContext ctx(heif_context_alloc());
  STD_TORCH_CHECK(ctx != nullptr, "Failed to allocate libheif context.");

  heif_error err = heif_context_read_from_memory_without_copy(
      ctx.get(),
      input.const_data_ptr<uint8_t>(),
      static_cast<size_t>(input.numel()),
      /*options=*/nullptr);
  STD_TORCH_CHECK(
      err.code == heif_error_Ok,
      "heif_context_read_from_memory_without_copy failed: ",
      err.message);

  // TODO: properly support (or error on) image sequences.
  heif_image_handle* raw_handle = nullptr;
  err = heif_context_get_primary_image_handle(ctx.get(), &raw_handle);
  STD_TORCH_CHECK(
      err.code == heif_error_Ok,
      "heif_context_get_primary_image_handle failed: ",
      err.message);
  UniqueHeifImageHandle handle(raw_handle);

  int bit_depth = heif_image_handle_get_luma_bits_per_pixel(handle.get());
  STD_TORCH_CHECK(
      bit_depth > 0, "Failed to get a valid bit depth from the HEIC image.");
  bool source_gt_8bit = bit_depth > 8;

  bool has_alpha =
      static_cast<bool>(heif_image_handle_has_alpha_channel(handle.get()));
  bool return_rgb =
      should_return_rgb(static_cast<ImageReadMode>(mode), has_alpha);
  int num_channels = return_rgb ? 3 : 4;

  // libheif doesn't support decoding a 10/12-bit image into uint8 (libavif
  // does!) or decoding an 8bit image into uint16. We decode into 8bit or uint16
  // for >8bit images, and run the necessary post-processing in Python to
  // respect the desired output_dtype.
  constexpr bool little_endian = std::endian::native == std::endian::little;
  heif_chroma chroma;
  if (source_gt_8bit) {
    if (return_rgb) {
      chroma = little_endian ? heif_chroma_interleaved_RRGGBB_LE
                             : heif_chroma_interleaved_RRGGBB_BE;
    } else {
      chroma = little_endian ? heif_chroma_interleaved_RRGGBBAA_LE
                             : heif_chroma_interleaved_RRGGBBAA_BE;
    }
  } else {
    chroma =
        return_rgb ? heif_chroma_interleaved_RGB : heif_chroma_interleaved_RGBA;
  }

  // libheif applies the image's 'irot'/'imir' transforms during decode by
  // default (heif_decoding_options.ignore_transformations == false), so the
  // output is already correctly oriented. We deliberately do NOT run our own
  // exif_orientation_transform() afterwards, to avoid double-applying.
  heif_image* raw_img = nullptr;
  err = heif_decode_image(
      handle.get(),
      &raw_img,
      heif_colorspace_RGB,
      chroma,
      /*options=*/nullptr);
  STD_TORCH_CHECK(
      err.code == heif_error_Ok,
      "heif_decode_image failed: ",
      err.message,
      ". If this is an \"Unsupported codec\" error, the libheif found at runtime "
      "was built/installed without a decoder for this image's codec (typically "
      "libde265 for HEVC-coded HEIC). Install a libheif with HEVC decode support "
      "(e.g. `conda install -c conda-forge libheif`, which pulls libde265).");
  UniqueHeifImage img(raw_img);

  int stride = 0;
  const uint8_t* decoded_data = heif_image_get_plane_readonly(
      img.get(), heif_channel_interleaved, &stride);
  STD_TORCH_CHECK(
      decoded_data != nullptr, "Failed to get the decoded HEIC image plane.");

  int64_t height = heif_image_handle_get_height(handle.get());
  int64_t width = heif_image_handle_get_width(handle.get());

  // Allocate an (H, W, C) contiguous tensor and copy the decoded plane into it
  // row by row: the plane's `stride` may include per-row padding, and the
  // buffer is owned by `img` (freed when it goes out of scope), so we can't
  // wrap it with from_blob.
  torch::stable::Tensor output = torch::stable::empty(
      {height, width, static_cast<int64_t>(num_channels)},
      source_gt_8bit ? kStableUInt16 : kStableUInt8);
  auto* output_ptr = static_cast<uint8_t*>(output.mutable_data_ptr());

  int64_t row_num_bytes = width * num_channels * (source_gt_8bit ? 2 : 1);
  for (int64_t h = 0; h < height; ++h) {
    std::memcpy(
        output_ptr + h * row_num_bytes,
        decoded_data + h * stride,
        static_cast<size_t>(row_num_bytes));
  }

  if (source_gt_8bit) {
    // 10bit and 12bit images are decoded into uint16 but are not scaled, so we
    // do the scaling ourselves below into the uint16 range - that's what our
    // Python side expects.

    auto* output_ptr_16 = reinterpret_cast<uint16_t*>(output_ptr);
    int64_t num_values = height * width * num_channels;
    for (int64_t p = 0; p < num_values; ++p) {
      output_ptr_16[p] <<= (16 - bit_depth);
    }
  }

  return stable_permute(output, {2, 0, 1});
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_HEIC
