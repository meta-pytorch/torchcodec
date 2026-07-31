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
    [[maybe_unused]] int64_t mode,
    [[maybe_unused]] int64_t output_dtype) {
  STD_TORCH_CHECK(
      false,
      "decode_heic: torchcodec was not compiled with libheif support. Rebuild "
      "torchcodec with TORCHCODEC_BUILD_HEIC=1. If you see this error in a "
      "prebuilt wheel, please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

#include <torch/headeronly/core/MemoryFormat.h>

#include <bit>
#include <cstring>
#include <memory>
#include <vector>

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
    int64_t mode,
    int64_t output_dtype) {
  auto contig_input = validate_encoded_data(input);

  UniqueHeifContext ctx(heif_context_alloc());
  STD_TORCH_CHECK(ctx != nullptr, "Failed to allocate libheif context.");

  heif_error err = heif_context_read_from_memory_without_copy(
      ctx.get(),
      contig_input.const_data_ptr<uint8_t>(),
      static_cast<size_t>(contig_input.numel()),
      /*options=*/nullptr);
  STD_TORCH_CHECK(
      err.code == heif_error_Ok,
      "heif_context_read_from_memory_without_copy failed: ",
      err.message);

  // A HEIF file can hold several "top-level" images (an image sequence /
  // burst). We decode all of them into a batched (N, C, H, W) tensor, one per
  // top-level image, and squeeze the batch dim below when there's only one.
  // Thumbnails and grid tiles are not top-level images, so they don't show up
  // here.
  int num_images = heif_context_get_number_of_top_level_images(ctx.get());
  STD_TORCH_CHECK(
      num_images > 0, "HEIC file should contain at least one top-level image.");

  std::vector<heif_item_id> image_ids(static_cast<size_t>(num_images));
  int num_ids = heif_context_get_list_of_top_level_image_IDs(
      ctx.get(), image_ids.data(), num_images);
  STD_TORCH_CHECK(
      num_ids == num_images,
      "Expected ",
      num_images,
      " top-level HEIC image IDs but got ",
      num_ids,
      ".");

  // Geometry/format is derived from the first image and every subsequent frame
  // is required to match it (validated below), so the whole sequence fits a
  // single output tensor.
  torch::stable::Tensor output;
  uint8_t* output_ptr = nullptr;
  int64_t frame_num_bytes = 0;
  int64_t height = 0;
  int64_t width = 0;
  int num_channels = 0;
  int bit_depth = 0;
  bool decode_16 = false;
  heif_chroma chroma = heif_chroma_undefined;

  for (int64_t i = 0; i < num_images; ++i) {
    heif_image_handle* raw_handle = nullptr;
    err = heif_context_get_image_handle(
        ctx.get(), image_ids[static_cast<size_t>(i)], &raw_handle);
    STD_TORCH_CHECK(
        err.code == heif_error_Ok,
        "heif_context_get_image_handle failed at frame ",
        i,
        ": ",
        err.message);
    UniqueHeifImageHandle handle(raw_handle);

    int64_t frame_height = heif_image_handle_get_height(handle.get());
    int64_t frame_width = heif_image_handle_get_width(handle.get());
    int frame_bit_depth =
        heif_image_handle_get_luma_bits_per_pixel(handle.get());
    STD_TORCH_CHECK(
        frame_bit_depth > 0,
        "Failed to get a valid bit depth from the HEIC image.");
    bool frame_has_alpha =
        static_cast<bool>(heif_image_handle_has_alpha_channel(handle.get()));

    if (i == 0) {
      height = frame_height;
      width = frame_width;
      bit_depth = frame_bit_depth;
      bool source_gt_8bit = bit_depth > 8;

      bool return_rgb =
          should_return_rgb(static_cast<ImageReadMode>(mode), frame_has_alpha);
      num_channels = return_rgb ? 3 : 4;

      // Decode into a 16-bit container only when the source actually carries >8
      // bits AND 16-bit output is wanted. For uint8 output we let libheif
      // downscale a >8-bit source straight to 8-bit. We never ask libheif to
      // *widen* an 8-bit source to 16 bits: that would be a lossy 8->10->16
      // hop, so 8->16 is done exactly as `* 257` in Python instead.
      bool output_16 = should_output_uint16(
          static_cast<OutputDtype>(output_dtype), source_gt_8bit);
      decode_16 = source_gt_8bit && output_16;

      constexpr bool little_endian = std::endian::native == std::endian::little;
      if (decode_16) {
        if (return_rgb) {
          chroma = little_endian ? heif_chroma_interleaved_RRGGBB_LE
                                 : heif_chroma_interleaved_RRGGBB_BE;
        } else {
          chroma = little_endian ? heif_chroma_interleaved_RRGGBBAA_LE
                                 : heif_chroma_interleaved_RRGGBBAA_BE;
        }
      } else {
        chroma = return_rgb ? heif_chroma_interleaved_RGB
                            : heif_chroma_interleaved_RGBA;
      }

      // ChannelsLast so each frame's region is laid out as contiguous
      // interleaved HWC bytes, matching what libheif hands us below.
      output = torch::stable::empty(
          {static_cast<int64_t>(num_images), num_channels, height, width},
          decode_16 ? kStableUInt16 : kStableUInt8,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          torch::headeronly::MemoryFormat::ChannelsLast);
      output_ptr = static_cast<uint8_t*>(output.mutable_data_ptr());
      frame_num_bytes = num_channels * height * width * (decode_16 ? 2 : 1);
    } else {
      STD_TORCH_CHECK(
          frame_height == height && frame_width == width &&
              frame_bit_depth == bit_depth,
          "HEIC image sequence has frames with mismatched geometry: frame ",
          i,
          " is ",
          frame_width,
          "x",
          frame_height,
          " at ",
          frame_bit_depth,
          " bits vs ",
          width,
          "x",
          height,
          " at ",
          bit_depth,
          " bits for the first frame. This is not supported.");
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
        "heif_decode_image failed at frame ",
        i,
        ": ",
        err.message,
        ". If this is an \"Unsupported codec\" error, the libheif found at "
        "runtime was built/installed without a decoder for this image's codec "
        "(typically libde265 for HEVC-coded HEIC). Install a libheif with HEVC "
        "decode support (e.g. `conda install -c conda-forge libheif`, which "
        "pulls libde265).");
    UniqueHeifImage img(raw_img);

    int stride = 0;
    const uint8_t* decoded_data = heif_image_get_plane_readonly(
        img.get(), heif_channel_interleaved, &stride);
    STD_TORCH_CHECK(
        decoded_data != nullptr, "Failed to get the decoded HEIC image plane.");

    // Copy the decoded plane into this frame's region row by row: the plane's
    // `stride` may include per-row padding, and the buffer is owned by `img`
    // (freed when it goes out of scope), so we can't wrap it with from_blob.
    uint8_t* frame_ptr = output_ptr + i * frame_num_bytes;
    int64_t row_num_bytes = width * num_channels * (decode_16 ? 2 : 1);
    for (int64_t h = 0; h < height; ++h) {
      std::memcpy(
          frame_ptr + h * row_num_bytes,
          decoded_data + h * stride,
          static_cast<size_t>(row_num_bytes));
    }
  }

  if (decode_16) {
    // libheif writes the decoded values at their native bit depth (e.g. in
    // [0, 1023] for 10-bit), NOT scaled to the full uint16 range, so we expand
    // them ourselves.
    STD_TORCH_CHECK(
        bit_depth <= 16,
        "Unexpected HEIC bit depth greater than 16: ",
        bit_depth);
    int shift = 16 - bit_depth;
    auto* output_ptr_16 = reinterpret_cast<uint16_t*>(output_ptr);
    int64_t num_values = num_images * height * width * num_channels;
    for (int64_t p = 0; p < num_values; ++p) {
      uint16_t v = output_ptr_16[p];
      output_ptr_16[p] =
          static_cast<uint16_t>((v << shift) | (v >> (bit_depth - shift)));
    }
  }

  if (num_images == 1) {
    output = select_row(output, 0);
  }
  return output;
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_HEIC
