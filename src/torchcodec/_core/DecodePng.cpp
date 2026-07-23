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

#if !TORCHCODEC_ENABLE_PNG

namespace facebook::torchcodec {

torch::stable::Tensor decode_png(
    [[maybe_unused]] const torch::stable::Tensor& data,
    [[maybe_unused]] int64_t mode,
    [[maybe_unused]] int64_t output_dtype) {
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
#include <bit>
#include <cstdint>
#include <cstdio>

#include "Exif.h"
#include "ImageCommon.h"

namespace facebook::torchcodec {

using namespace exif_private;

namespace {

struct ErrorCtx {
  char error_message[256] = "";
};

void error_callback(png_structp png_ptr, png_const_charp error_message) {
  auto* error_ctx = static_cast<ErrorCtx*>(png_get_error_ptr(png_ptr));
  if (error_ctx != nullptr) {
    std::snprintf(
        error_ctx->error_message,
        sizeof(error_ctx->error_message),
        "%s",
        error_message);
  }
  png_longjmp(png_ptr, 1);
}

ExifOrientation fetch_png_exif_orientation(
    png_structp png_ptr,
    png_infop info_ptr) {
  png_uint_32 num_exif = 0;
  png_bytep exif = nullptr;

  if (png_get_valid(png_ptr, info_ptr, PNG_INFO_eXIf)) {
    png_get_eXIf_1(png_ptr, info_ptr, &num_exif, &exif);
  }

  if (exif != nullptr && num_exif > 0) {
    return fetch_exif_orientation(exif, num_exif);
  } else {
    return ExifOrientation::Unspecified;
  }
}

struct SourceCtx {
  png_const_bytep ptr;
  png_size_t count;
};

void read_callback(png_structp png_ptr, png_bytep output, png_size_t bytes) {
  auto* source_ctx = static_cast<SourceCtx*>(png_get_io_ptr(png_ptr));
  if (source_ctx->count < bytes) {
    // trigger our error callback
    png_error(
        png_ptr,
        "Out of bound read in decode_png. The input image might be corrupted?");
  }
  std::copy(source_ctx->ptr, source_ctx->ptr + bytes, output);
  source_ctx->ptr += bytes;
  source_ctx->count -= bytes;
}

struct PngHeader {
  png_uint_32 width;
  png_uint_32 height;
  int num_output_channels;
  bool output_16;
  int num_passes;
};

PngHeader read_header_and_configure(
    png_structp& png_ptr,
    png_infop& info_ptr,
    ErrorCtx& error_ctx,
    SourceCtx& source_ctx,
    ImageReadMode read_mode,
    OutputDtype output_dtype) {
  if (setjmp(png_jmpbuf(png_ptr)) != 0) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    STD_TORCH_CHECK(false, "decode_png failed: ", error_ctx.error_message);
  }

  png_set_read_fn(png_ptr, &source_ctx, read_callback);
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
    STD_TORCH_CHECK(retval == 1, "Could not read image metadata from content.")
  }

  if (bit_depth > 8 && bit_depth != 16) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    STD_TORCH_CHECK(
        false,
        "bit depth of png image is ",
        bit_depth,
        ". Only <=8 and 16 are supported.")
  }

  bool output_16 = should_output_uint16(output_dtype, bit_depth == 16);

  int num_output_channels = png_get_channels(png_ptr, info_ptr);

  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
    png_set_expand_gray_1_2_4_to_8(png_ptr);
  }

  int num_passes;
  if (interlace_type == PNG_INTERLACE_ADAM7) {
    num_passes = png_set_interlace_handling(png_ptr);
  } else {
    num_passes = 1;
  }

  if (read_mode != ImageReadMode::UNCHANGED) {
    bool is_palette = (color_type & PNG_COLOR_MASK_PALETTE) != 0;
    bool has_color = (color_type & PNG_COLOR_MASK_COLOR) != 0;
    bool has_alpha = (color_type & PNG_COLOR_MASK_ALPHA) != 0;
    // A tRNS chunk encodes transparency without a dedicated alpha channel.
    // png_set_tRNS_to_alpha() expands it into a real alpha channel (it must be
    // called after png_set_palette_to_rgb()).
    bool has_trns = png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS) != 0;

    png_uint_32 opaque_alpha = output_16 ? 65535 : 255;

    switch (read_mode) {
      case ImageReadMode::GRAY:
        if (color_type != PNG_COLOR_TYPE_GRAY) {
          if (is_palette) {
            png_set_palette_to_rgb(png_ptr);
          }

          if (has_alpha) {
            png_set_strip_alpha(png_ptr);
          }

          if (has_color) {
            png_set_rgb_to_gray(png_ptr, 1, 0.2989, 0.587);
          }
          num_output_channels = 1;
        }
        break;
      case ImageReadMode::GRAY_ALPHA:
        if (color_type != PNG_COLOR_TYPE_GRAY_ALPHA) {
          if (is_palette) {
            png_set_palette_to_rgb(png_ptr);
          }

          if (has_trns) {
            png_set_tRNS_to_alpha(png_ptr);
          } else if (!has_alpha) {
            png_set_add_alpha(png_ptr, opaque_alpha, PNG_FILLER_AFTER);
          }

          if (has_color) {
            png_set_rgb_to_gray(png_ptr, 1, 0.2989, 0.587);
          }
          num_output_channels = 2;
        }
        break;
      case ImageReadMode::RGB:
        if (color_type != PNG_COLOR_TYPE_RGB) {
          if (is_palette) {
            png_set_palette_to_rgb(png_ptr);
          } else if (!has_color) {
            png_set_gray_to_rgb(png_ptr);
          }

          if (has_alpha) {
            png_set_strip_alpha(png_ptr);
          }
          num_output_channels = 3;
        }
        break;
      case ImageReadMode::RGB_ALPHA:
        if (color_type != PNG_COLOR_TYPE_RGB_ALPHA) {
          if (is_palette) {
            png_set_palette_to_rgb(png_ptr);
          } else if (!has_color) {
            png_set_gray_to_rgb(png_ptr);
          }

          if (has_trns) {
            png_set_tRNS_to_alpha(png_ptr);
          } else if (!has_alpha) {
            png_set_add_alpha(png_ptr, opaque_alpha, PNG_FILLER_AFTER);
          }
          num_output_channels = 4;
        }
        break;
      default:
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        STD_TORCH_CHECK(
            false,
            "Reached an unexpected code path while decoding a PNG file to mode ",
            static_cast<int64_t>(read_mode),
            ". This should never happen, please report a bug to the TorchCodec repo.");
    }
  }

  // libpng knows how to scale 16-bit samples to 8-bit, and vice versa, so we
  // can use that instead of doing it ourselves.
  bool need_scaling = false;
  if (output_16 && bit_depth != 16) {
    png_set_expand_16(png_ptr);
    need_scaling = true;
  } else if (!output_16 && bit_depth == 16) {
    png_set_scale_16(png_ptr);
    need_scaling = true;
  }

  if (read_mode != ImageReadMode::UNCHANGED || need_scaling) {
    png_read_update_info(png_ptr, info_ptr);
  }

  return {
      .width = width,
      .height = height,
      .num_output_channels = num_output_channels,
      .output_16 = output_16,
      .num_passes = num_passes};
}

void decode_rows(
    png_structp& png_ptr,
    png_infop& info_ptr,
    ErrorCtx& error_ctx,
    uint8_t* output_ptr,
    png_uint_32 height,
    int num_passes,
    int64_t stride) {
  if (setjmp(png_jmpbuf(png_ptr)) != 0) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    STD_TORCH_CHECK(false, "decode_png failed: ", error_ctx.error_message);
  }

  if (std::endian::native == std::endian::little) {
    png_set_swap(png_ptr);
  }

  for (int pass = 0; pass < num_passes; pass++) {
    uint8_t* row_ptr = output_ptr;
    for (png_uint_32 i = 0; i < height; ++i) {
      png_read_row(png_ptr, row_ptr, nullptr);
      row_ptr += stride;
    }
  }
}

} // namespace

// Important: see the [libjpeg error handling] in the jpeg decoder: everything
// applies here too. Critically, we must not throw a C++ exception through
// libpng's C stack (and callbacks), and we also shouldn't allocate anything
// that needs proper destruction in a function that defines a setjmp() point.
// This is why the output tensor is allocated here in decode_png() and
// decode_png() does *not* define a setjmp() point.

torch::stable::Tensor decode_png(
    const torch::stable::Tensor& input,
    int64_t mode,
    int64_t output_dtype) {
  validate_encoded_data(input);

  auto input_ptr = input.const_data_ptr<uint8_t>();

  auto png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  STD_TORCH_CHECK(
      png_ptr != nullptr, "libpng read structure allocation failed!")
  auto info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == nullptr) {
    png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    STD_TORCH_CHECK(info_ptr, "libpng info structure allocation failed!")
  }

  ErrorCtx error_ctx;
  png_set_error_fn(png_ptr, &error_ctx, error_callback, /*warn_fn=*/nullptr);

  SourceCtx source_ctx;
  source_ctx.ptr = reinterpret_cast<png_const_bytep>(input_ptr);
  source_ctx.count = input.numel();

  torch::stable::Tensor output;

  auto header = read_header_and_configure(
      png_ptr,
      info_ptr,
      error_ctx,
      source_ctx,
      static_cast<ImageReadMode>(mode),
      static_cast<OutputDtype>(output_dtype));

  auto output_16 = header.output_16;
  output = torch::stable::empty(
      {static_cast<int64_t>(header.height),
       static_cast<int64_t>(header.width),
       header.num_output_channels},
      output_16 ? kStableUInt16 : kStableUInt8);

  int64_t bytes_per_pixel = header.num_output_channels * (output_16 ? 2 : 1);
  int64_t stride = static_cast<int64_t>(header.width) * bytes_per_pixel;
  decode_rows(
      png_ptr,
      info_ptr,
      error_ctx,
      static_cast<uint8_t*>(output.mutable_data_ptr()),
      header.height,
      header.num_passes,
      stride);

  ExifOrientation exif_orientation =
      fetch_png_exif_orientation(png_ptr, info_ptr);

  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);

  return exif_orientation_transform(
      stable_permute(output, {2, 0, 1}), exif_orientation);
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_PNG
