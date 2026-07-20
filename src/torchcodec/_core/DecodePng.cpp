// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecodePng.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !PNG_FOUND

namespace facebook::torchcodec {

torch::stable::Tensor decode_png(
    const torch::stable::Tensor& data,
    int64_t mode) {
  STD_TORCH_CHECK(
      false,
      "decode_png: torchcodec was not compiled with libpng support. "
      "Rebuild torchcodec in an environment where libpng (and its development "
      "headers) are available. If you see this error in a prebuilt wheel, "
      "please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

#include <png.h>

#if !defined(PNG_eXIf_SUPPORTED)
// We always want to apply exif so we enforce a libpng that has eXIf support.
// This is a compile-time check, not a runtime check, but since we bundle
// libpng within our wheel this exif support is guaranteed to users.
#error \
    "torchcodec requires libpng to be built with eXIf support (PNG_eXIf_SUPPORTED)."
#endif

#include <setjmp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

#include "Exif.h"
#include "ImageCommon.h"

namespace facebook::torchcodec {

using namespace exif_private;

namespace {

bool is_little_endian() {
  uint32_t x = 1;
  return *(uint8_t*)&x;
}

struct PngErrorContext {
  char error_message[256] = "";
};

void png_error_callback(png_structp png_ptr, png_const_charp error_message) {
  auto* error_ctx = static_cast<PngErrorContext*>(png_get_error_ptr(png_ptr));
  if (error_ctx != nullptr) {
    std::snprintf(
        error_ctx->error_message,
        sizeof(error_ctx->error_message),
        "%s",
        error_message);
  }
  png_longjmp(png_ptr, 1);
}

int fetch_png_exif_orientation(png_structp png_ptr, png_infop info_ptr) {
  png_uint_32 num_exif = 0;
  png_bytep exif = 0;

  // Exif info could be in info_ptr
  if (png_get_valid(png_ptr, info_ptr, PNG_INFO_eXIf)) {
    png_get_eXIf_1(png_ptr, info_ptr, &num_exif, &exif);
  }

  if (exif && num_exif > 0) {
    return fetch_exif_orientation(exif, num_exif);
  }
  return -1;
}

} // namespace

torch::stable::Tensor decode_png(
    const torch::stable::Tensor& data,
    int64_t mode) {
  validate_encoded_data(data);

  auto png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  STD_TORCH_CHECK(png_ptr, "libpng read structure allocation failed!")
  auto info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    // Seems redundant with the if statement. done here to avoid leaking memory.
    STD_TORCH_CHECK(info_ptr, "libpng info structure allocation failed!")
  }

  auto datap = data.const_data_ptr<uint8_t>();
  auto datap_len = data.numel();

  // Capture libpng's error message (see png_error_callback) so we can report it
  // instead of a generic "internal error".
  PngErrorContext error_ctx;
  png_set_error_fn(
      png_ptr, &error_ctx, png_error_callback, /*warn_fn=*/nullptr);

  // NOTE: libpng uses setjmp/longjmp for error handling. longjmp does not
  // unwind C++ stack frames, so destructors of objects created after setjmp
  // won't run. We use std::optional to declare tensors before setjmp while
  // deferring construction, and explicitly reset them on the error path.
  std::optional<torch::stable::Tensor> tensor;

  if (setjmp(png_jmpbuf(png_ptr)) != 0) {
    tensor.reset();
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    STD_TORCH_CHECK(false, "decode_png failed: ", error_ctx.error_message);
  }
  STD_TORCH_CHECK(datap_len >= 8, "Content is too small for png!")
  auto is_png = !png_sig_cmp(datap, 0, 8);
  STD_TORCH_CHECK(is_png, "Content is not png!")

  struct Reader {
    png_const_bytep ptr;
    png_size_t count;
  } reader;

  reader.ptr = png_const_bytep(datap) + 8;
  reader.count = datap_len - 8;

  auto read_callback = [](png_structp png_ptr,
                          png_bytep output,
                          png_size_t bytes) {
    auto reader = static_cast<Reader*>(png_get_io_ptr(png_ptr));
    STD_TORCH_CHECK(
        reader->count >= bytes,
        "Out of bound read in decode_png. The input image might be corrupted?");
    std::copy(reader->ptr, reader->ptr + bytes, output);
    reader->ptr += bytes;
    reader->count -= bytes;
  };
  png_set_sig_bytes(png_ptr, 8);
  png_set_read_fn(png_ptr, &reader, read_callback);
  png_read_info(png_ptr, info_ptr);

  png_uint_32 width, height;
  int bit_depth, color_type;
  int interlace_type;
  auto retval = png_get_IHDR(
      png_ptr,
      info_ptr,
      &width,
      &height,
      &bit_depth,
      &color_type,
      &interlace_type,
      nullptr,
      nullptr);

  if (retval != 1) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    STD_TORCH_CHECK(retval == 1, "Could read image metadata from content.")
  }

  if (bit_depth > 8 && bit_depth != 16) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    STD_TORCH_CHECK(
        false,
        "bit depth of png image is " + std::to_string(bit_depth) +
            ". Only <=8 and 16 are supported.")
  }

  int channels = png_get_channels(png_ptr, info_ptr);

  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
    png_set_expand_gray_1_2_4_to_8(png_ptr);
  }

  int number_of_passes;
  if (interlace_type == PNG_INTERLACE_ADAM7) {
    number_of_passes = png_set_interlace_handling(png_ptr);
  } else {
    number_of_passes = 1;
  }

  auto read_mode = static_cast<ImageReadMode>(mode);
  if (read_mode != ImageReadMode::Unchanged) {
    // TODO: consider supporting PNG_INFO_tRNS
    bool is_palette = (color_type & PNG_COLOR_MASK_PALETTE) != 0;
    bool has_color = (color_type & PNG_COLOR_MASK_COLOR) != 0;
    bool has_alpha = (color_type & PNG_COLOR_MASK_ALPHA) != 0;

    switch (read_mode) {
      case ImageReadMode::Gray:
        if (color_type != PNG_COLOR_TYPE_GRAY) {
          if (is_palette) {
            png_set_palette_to_rgb(png_ptr);
            has_alpha = true;
          }

          if (has_alpha) {
            png_set_strip_alpha(png_ptr);
          }

          if (has_color) {
            png_set_rgb_to_gray(png_ptr, 1, 0.2989, 0.587);
          }
          channels = 1;
        }
        break;
      case ImageReadMode::GrayAlpha:
        if (color_type != PNG_COLOR_TYPE_GRAY_ALPHA) {
          if (is_palette) {
            png_set_palette_to_rgb(png_ptr);
            has_alpha = true;
          }

          if (!has_alpha) {
            png_set_add_alpha(png_ptr, (1 << bit_depth) - 1, PNG_FILLER_AFTER);
          }

          if (has_color) {
            png_set_rgb_to_gray(png_ptr, 1, 0.2989, 0.587);
          }
          channels = 2;
        }
        break;
      case ImageReadMode::Rgb:
        if (color_type != PNG_COLOR_TYPE_RGB) {
          if (is_palette) {
            png_set_palette_to_rgb(png_ptr);
            has_alpha = true;
          } else if (!has_color) {
            png_set_gray_to_rgb(png_ptr);
          }

          if (has_alpha) {
            png_set_strip_alpha(png_ptr);
          }
          channels = 3;
        }
        break;
      case ImageReadMode::RgbAlpha:
        if (color_type != PNG_COLOR_TYPE_RGB_ALPHA) {
          if (is_palette) {
            png_set_palette_to_rgb(png_ptr);
            has_alpha = true;
          } else if (!has_color) {
            png_set_gray_to_rgb(png_ptr);
          }

          if (!has_alpha) {
            png_set_add_alpha(png_ptr, (1 << bit_depth) - 1, PNG_FILLER_AFTER);
          }
          channels = 4;
        }
        break;
      default:
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        STD_TORCH_CHECK(
            false, "The provided mode is not supported for PNG files");
    }

    png_read_update_info(png_ptr, info_ptr);
  }

  auto num_pixels_per_row = width * channels;
  auto is_16_bits = bit_depth == 16;
  tensor = torch::stable::empty(
      {int64_t(height), int64_t(width), channels},
      is_16_bits ? kStableUInt16 : kStableUInt8);
  if (is_little_endian()) {
    png_set_swap(png_ptr);
  }
  auto t_ptr = (uint8_t*)tensor->mutable_data_ptr();
  for (int pass = 0; pass < number_of_passes; pass++) {
    for (png_uint_32 i = 0; i < height; ++i) {
      png_read_row(png_ptr, t_ptr, nullptr);
      t_ptr += num_pixels_per_row * (is_16_bits ? 2 : 1);
    }
    t_ptr = (uint8_t*)tensor->mutable_data_ptr();
  }

  // torchcodec always applies EXIF orientation (unlike torchvision, which gates
  // this behind an apply_exif_orientation parameter).
  int exif_orientation = fetch_png_exif_orientation(png_ptr, info_ptr);

  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);

  return exif_orientation_transform(
      stable_permute(*tensor, {2, 0, 1}), exif_orientation);
}

} // namespace facebook::torchcodec

#endif // !PNG_FOUND
