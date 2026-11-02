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

#include <cstring>

#ifdef _WIN32
#include <fstream>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "ImageCommon.h"

namespace facebook::torchcodec {

namespace {

int64_t read_chunk_length(const uint8_t* header) {
  return (int64_t(header[0]) << 24) | (int64_t(header[1]) << 16) |
      (int64_t(header[2]) << 8) | int64_t(header[3]);
}

} // namespace

int64_t png_skip_idat_chunks_from_tensor(
    const torch::stable::Tensor& input,
    int64_t offset) {
  auto contig_input = validate_encoded_data(input);
  auto data = contig_input.const_data_ptr<uint8_t>();
  int64_t size = contig_input.numel();
  STD_TORCH_CHECK(offset >= 0, "offset must be non-negative, got ", offset);
  while (offset + 8 <= size && std::memcmp(data + offset + 4, "IDAT", 4) == 0) {
    offset += 12 + read_chunk_length(data + offset);
  }
  return offset;
}

int64_t png_skip_idat_chunks_from_file(std::string filename, int64_t offset) {
  STD_TORCH_CHECK(offset >= 0, "offset must be non-negative, got ", offset);
  // The chunk headers are usually 8 KB apart (libpng's default), so buffered
  // reads would end up reading the whole file: we do one small unbuffered read
  // per chunk instead.
  uint8_t header[8];
#ifdef _WIN32
  std::ifstream file;
  file.rdbuf()->pubsetbuf(nullptr, 0);
  file.open(filename, std::ios::binary);
  STD_TORCH_CHECK(
      file.is_open(), "Failed to open file for reading: ", filename);
  while (file.seekg(offset) &&
         file.read(reinterpret_cast<char*>(header), sizeof(header)) &&
         std::memcmp(header + 4, "IDAT", 4) == 0) {
    offset += 12 + read_chunk_length(header);
  }
#else
  int fd = open(filename.c_str(), O_RDONLY | O_CLOEXEC);
  STD_TORCH_CHECK(fd >= 0, "Failed to open file for reading: ", filename);
  while (pread(fd, header, sizeof(header), offset) ==
             static_cast<ssize_t>(sizeof(header)) &&
         std::memcmp(header + 4, "IDAT", 4) == 0) {
    offset += 12 + read_chunk_length(header);
  }
  close(fd);
#endif
  return offset;
}

} // namespace facebook::torchcodec

#if !TORCHCODEC_ENABLE_PNG

namespace facebook::torchcodec {

torch::stable::Tensor decode_png(
    [[maybe_unused]] const torch::stable::Tensor& input,
    [[maybe_unused]] int64_t mode,
    [[maybe_unused]] int64_t output_dtype) {
  STD_TORCH_CHECK(
      false,
      "decode_png: torchcodec was not compiled with libpng support. Rebuild "
      "torchcodec with TORCHCODEC_BUILD_PNG=1. If you see this error in a "
      "prebuilt wheel, please report it to the TorchCodec repo.");
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
#include <cstring>
#include <string_view>
#include <vector>

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

// Decodes the text of an ImageMagick "Raw profile type exif" chunk, i.e.
// "\nexif\n<length>\n<hex data, over multiple lines>", the way Pillow does:
// bytes.fromhex("".join(text.split("\n")[3:])). Returns false if the hex data
// is invalid.
bool decode_raw_profile_exif(
    std::string_view text,
    std::vector<unsigned char>& exif) {
  for (int i = 0; i < 3; ++i) {
    size_t newline = text.find('\n');
    if (newline == std::string_view::npos) {
      return true;
    }
    text.remove_prefix(newline + 1);
  }
  auto hex_value = [](char c) -> int {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
      return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
      return c - 'A' + 10;
    }
    return -1;
  };
  int high_nibble = -1;
  for (char c : text) {
    if (c == '\n') {
      continue;
    }
    if (high_nibble < 0 &&
        std::string_view(" \t\r\v\f").find(c) != std::string_view::npos) {
      continue;
    }
    int value = hex_value(c);
    if (value < 0) {
      return false;
    }
    if (high_nibble < 0) {
      high_nibble = value;
    } else {
      exif.push_back(static_cast<unsigned char>(high_nibble << 4 | value));
      high_nibble = -1;
    }
  }
  return high_nibble < 0;
}

// Finds the orientation the way Pillow's ImageOps.exif_transpose() does, after
// it loaded the whole file: from the eXIf chunk, or else from an ImageMagick
// "Raw profile type exif" text chunk, and if the EXIF data has no orientation,
// from the XMP text chunk. All of these can come before or after the image
// data, i.e. in info_ptr or in end_info.
ExifOrientation fetch_png_orientation(
    png_structp png_ptr,
    png_infop info_ptr,
    png_infop end_info) {
  png_bytep exif = nullptr;
  png_uint_32 num_exif = 0;
  std::string_view raw_profile_exif;
  std::string_view xmp;
  bool found_raw_profile_exif = false;
  bool found_xmp = false;
  for (png_infop info : {info_ptr, end_info}) {
    if (info == nullptr) {
      continue;
    }
    if (exif == nullptr && png_get_valid(png_ptr, info, PNG_INFO_eXIf)) {
      png_get_eXIf_1(png_ptr, info, &num_exif, &exif);
    }
    png_textp texts = nullptr;
    int num_texts = png_get_text(png_ptr, info, &texts, nullptr);
    for (int i = 0; i < num_texts; ++i) {
      std::string_view text(
          texts[i].text,
          texts[i].compression >= PNG_ITXT_COMPRESSION_NONE
              ? texts[i].itxt_length
              : texts[i].text_length);
      // Like Pillow, we keep the last chunk for each keyword.
      if (std::strcmp(texts[i].key, "Raw profile type exif") == 0) {
        raw_profile_exif = text;
        found_raw_profile_exif = true;
      } else if (std::strcmp(texts[i].key, "XML:com.adobe.xmp") == 0) {
        xmp = text;
        found_xmp = true;
      }
    }
  }

  ExifOrientation orientation = ExifOrientation::Unspecified;
  std::vector<unsigned char> decoded_exif;
  if (exif != nullptr && num_exif > 0) {
    orientation = fetch_exif_orientation(exif, num_exif);
  } else if (
      found_raw_profile_exif &&
      decode_raw_profile_exif(raw_profile_exif, decoded_exif)) {
    orientation = fetch_exif_orientation_after_prefixes(
        decoded_exif.data(), decoded_exif.size());
  }
  if (orientation == ExifOrientation::Unspecified && found_xmp) {
    orientation = fetch_xmp_orientation(
        reinterpret_cast<const unsigned char*>(xmp.data()), xmp.size());
  }
  return orientation;
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

  bool is_palette = (color_type & PNG_COLOR_MASK_PALETTE) != 0;
  // A tRNS chunk encodes transparency without a dedicated alpha channel.
  // png_set_tRNS_to_alpha() expands it into a real alpha channel (it must be
  // called after png_set_palette_to_rgb()).
  bool has_trns = png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS) != 0;

  if (read_mode == ImageReadMode::UNCHANGED) {
    if (is_palette) {
      // A PNG palette (PLTE chunk) is always RGB triplets
      // Note that this path is buggy in torchvision, as TV returns the raw
      // palette indices instead of the expanded RGB values. What we do here is
      // correct, and matches the behavior of PIL.
      png_set_palette_to_rgb(png_ptr);
      num_output_channels = 3;
      if (has_trns) {
        png_set_tRNS_to_alpha(png_ptr);
        num_output_channels = 4;
      }
    } else if (has_trns) {
      png_set_strip_alpha(png_ptr);
    }
  } else {
    bool has_color = (color_type & PNG_COLOR_MASK_COLOR) != 0;
    bool has_alpha = (color_type & PNG_COLOR_MASK_ALPHA) != 0 || has_trns;

    png_uint_32 opaque_alpha = output_16 ? 65535 : 255;

    switch (read_mode) {
      case ImageReadMode::GRAY:
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
        break;
      case ImageReadMode::GRAY_ALPHA:
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
        break;
      case ImageReadMode::RGB:
        if (is_palette) {
          png_set_palette_to_rgb(png_ptr);
        } else if (!has_color) {
          png_set_gray_to_rgb(png_ptr);
        }

        if (has_alpha) {
          png_set_strip_alpha(png_ptr);
        }
        num_output_channels = 3;
        break;
      case ImageReadMode::RGB_ALPHA:
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
  if (output_16 && bit_depth != 16) {
    png_set_expand_16(png_ptr);
  } else if (!output_16 && bit_depth == 16) {
    png_set_scale_16(png_ptr);
  }

  png_read_update_info(png_ptr, info_ptr);

  // png_read_row() writes as many bytes as libpng's own post-transformation
  // layout dictates, but we size the output tensor from num_output_channels,
  // which we track by hand above. We make sure they agree.
  size_t expected_row_bytes = static_cast<size_t>(width) *
      static_cast<size_t>(num_output_channels) * (output_16 ? 2 : 1);
  size_t actual_row_bytes = png_get_rowbytes(png_ptr, info_ptr);
  if (actual_row_bytes != expected_row_bytes) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    STD_TORCH_CHECK(
        false,
        "decode_png: libpng rows are ",
        actual_row_bytes,
        " bytes instead of the expected ",
        expected_row_bytes,
        ". This should never happen, please report a bug to the TorchCodec repo.");
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

// Reads the chunks after the image data, which can hold metadata too (into
// end_info). Unlike the image data, these are optional: on error, e.g. if the
// file is truncated after IDAT, we keep whatever chunks were read until then.
void read_end(png_structp png_ptr, png_infop end_info) {
  if (setjmp(png_jmpbuf(png_ptr)) != 0) {
    return;
  }
  png_read_end(png_ptr, end_info);
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
  auto contig_input = validate_encoded_data(input);

  auto input_ptr = contig_input.const_data_ptr<uint8_t>();

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
  source_ctx.count = contig_input.numel();

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

  auto end_info = png_create_info_struct(png_ptr);
  if (end_info != nullptr) {
    read_end(png_ptr, end_info);
  }
  ExifOrientation exif_orientation =
      fetch_png_orientation(png_ptr, info_ptr, end_info);

  png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);

  return exif_orientation_transform(
      stable_permute(output, {2, 0, 1}), exif_orientation);
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_PNG
