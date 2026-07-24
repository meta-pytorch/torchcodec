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

#include <cstring>
#include <memory>

#include "libheif/heif.h"

#include "ImageCommon.h"

namespace facebook::torchcodec {

namespace {

// RAII deleters for the libheif C-API handles, so we free resources on every
// path (including errors) without a try/catch. We use the C API rather than
// libheif/heif_cxx.h because the C++ wrapper isn't always installed and throws
// heif::Error across the op boundary, which surfaces to Python as "An unknown
// exception occurred".
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

using ContextPtr = std::unique_ptr<heif_context, HeifContextDeleter>;
using ImageHandlePtr =
    std::unique_ptr<heif_image_handle, HeifImageHandleDeleter>;
using ImagePtr = std::unique_ptr<heif_image, HeifImageDeleter>;

} // namespace

torch::stable::Tensor decode_heic(
    const torch::stable::Tensor& input,
    int64_t mode) {
  validate_encoded_data(input);

  ContextPtr ctx(heif_context_alloc());
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

  // TODO: properly support (or error on) image sequences. We only decode the
  // primary image, so multi-image HEIC files silently return just that one.
  // This is inconsistent with decode_gif (returns a batch) and decode_avif
  // (errors loudly), but libheif's get_number_of_top_level_images() doesn't
  // reliably tell sequences from grids/derived images (it disagrees with
  // libavif's imageCount), so we punt for now.
  heif_image_handle* raw_handle = nullptr;
  err = heif_context_get_primary_image_handle(ctx.get(), &raw_handle);
  STD_TORCH_CHECK(
      err.code == heif_error_Ok,
      "heif_context_get_primary_image_handle failed: ",
      err.message);
  ImageHandlePtr handle(raw_handle);

  int bit_depth = heif_image_handle_get_luma_bits_per_pixel(handle.get());
  STD_TORCH_CHECK(
      bit_depth > 0, "Failed to get a valid bit depth from the HEIC image.");
  bool source_gt_8bit = bit_depth > 8;

  bool has_alpha =
      static_cast<bool>(heif_image_handle_has_alpha_channel(handle.get()));
  bool return_rgb =
      should_return_rgb(static_cast<ImageReadMode>(mode), has_alpha);
  int num_channels = return_rgb ? 3 : 4;

  // We always decode at the source's NATIVE bit depth: 8-bit interleaved RGB(A)
  // for 8-bit sources, and little-endian 16-bit interleaved RGB(A)
  // (RRGGBB[AA]_LE) for >8-bit sources (remapped into the full uint16 range
  // below). Forcing a different output dtype (uint8/uint16) is done in Python.
  // Note: the _LE choice and the range remap may be wrong on big-endian
  // platforms (same caveat as the reference implementation).
  heif_chroma chroma;
  if (source_gt_8bit) {
    chroma = return_rgb ? heif_chroma_interleaved_RRGGBB_LE
                        : heif_chroma_interleaved_RRGGBBAA_LE;
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
  ImagePtr img(raw_img);

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
    // e.g. for a 10-bit source, decoded values are 10-bit numbers stored in
    // uint16. torchcodec/torchvision expect a uint16 value to span [0, 2**16),
    // so we left-shift by (16 - bit_depth) to map into the full range. (Other
    // libraries like libavif do this remap automatically; libheif does not.)
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
