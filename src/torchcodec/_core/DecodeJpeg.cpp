// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecodeJpeg.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !JPEG_FOUND

namespace facebook::torchcodec {

torch::stable::Tensor decode_jpeg(
    const torch::stable::Tensor& data,
    int64_t mode) {
  STD_TORCH_CHECK(
      false,
      "decode_jpeg: torchcodec was not compiled with libjpeg support. "
      "Rebuild torchcodec in an environment where libjpeg-turbo (and its "
      "development headers) are available. If you see this error in a prebuilt "
      "wheel, please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

#include <jpeglib.h>
#include <setjmp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>

#include "Exif.h"

namespace facebook::torchcodec {

using namespace exif_private;

namespace {

// Kept in-sync with the Python ImageColorMode enum in
// torchcodec/decoders/_image_decoders.py.
constexpr int64_t kImageReadModeUnchanged = 0;
constexpr int64_t kImageReadModeGray = 1;
constexpr int64_t kImageReadModeRgb = 3;

// EXIF APP1 marker: the orientation tag lives in this marker's payload.
constexpr uint16_t APP1 = 0xe1;

void validate_encoded_data(const torch::stable::Tensor& encoded_data) {
  STD_TORCH_CHECK(
      encoded_data.is_contiguous(), "Input tensor must be contiguous.");
  STD_TORCH_CHECK(
      encoded_data.scalar_type() == kStableUInt8,
      "Input tensor must have uint8 data type, got ",
      torch::headeronly::toString(encoded_data.scalar_type()));
  STD_TORCH_CHECK(
      encoded_data.dim() == 1 && encoded_data.numel() > 0,
      "Input tensor must be 1-dimensional and non-empty, got ",
      encoded_data.dim(),
      " dims  and ",
      encoded_data.numel(),
      " numels.");
}

constexpr JOCTET EOI_BUFFER[1] = {JPEG_EOI};

// The libjpeg base struct must come first: libjpeg only ever sees &base (via
// jpeg_ctx->err), and our callbacks cast that back to error_ctx_t* to reach the
// fields below.
struct error_ctx_t {
  jpeg_error_mgr base;
  char last_error_message[JMSG_LENGTH_MAX];
  jmp_buf setjmp_buffer; // for the longjmp back to decode_jpeg
};

void error_exit_cb(j_common_ptr jpeg_ctx) {
  // jpeg_ctx->err really points to an error_ctx_t struct, so coerce pointer.
  auto err = reinterpret_cast<error_ctx_t*>(jpeg_ctx->err);
  jpeg_ctx->err->format_message(jpeg_ctx, err->last_error_message);
  // Return control to the setjmp point.
  longjmp(err->setjmp_buffer, 1);
}

// Same base-first requirement as error_ctx_t: libjpeg sees &base (via
// jpeg_ctx->src), our callbacks cast it back to source_ctx_t*.
struct source_ctx_t {
  jpeg_source_mgr base;
  const JOCTET* data;
  size_t len;
};

boolean fill_input_buffer_cb(j_decompress_ptr jpeg_ctx) {
  // No more data.  Probably an incomplete image;  Raise exception.
  auto myerr = reinterpret_cast<error_ctx_t*>(jpeg_ctx->err);
  strcpy(myerr->last_error_message, "Image is incomplete or truncated");
  longjmp(myerr->setjmp_buffer, 1);
}

void skip_input_data_cb(j_decompress_ptr jpeg_ctx, long num_bytes) {
  auto* source_ctx = reinterpret_cast<source_ctx_t*>(jpeg_ctx->src);
  if (source_ctx->base.bytes_in_buffer < static_cast<size_t>(num_bytes)) {
    // Skipping over all of remaining data;  output EOI.
    source_ctx->base.next_input_byte = EOI_BUFFER;
    source_ctx->base.bytes_in_buffer = 1;
  } else {
    // Skipping over only some of the remaining data.
    source_ctx->base.next_input_byte += num_bytes;
    source_ctx->base.bytes_in_buffer -= num_bytes;
  }
}

void init_source_cb(j_decompress_ptr) {}

void term_source_cb(j_decompress_ptr) {}

void set_source_ctx(
    jpeg_decompress_struct& jpeg_ctx,
    const uint8_t* data,
    size_t len) {
  // We decode one image per fresh jpeg_decompress_struct, so jpeg_ctx.src is
  // always null here. Allocate our source manager from libjpeg's pool, which
  // jpeg_destroy_decompress frees.
  jpeg_ctx.src = static_cast<jpeg_source_mgr*>(jpeg_ctx.mem->alloc_small(
      reinterpret_cast<j_common_ptr>(&jpeg_ctx),
      JPOOL_PERMANENT,
      sizeof(source_ctx_t)));
  auto* source_ctx = reinterpret_cast<source_ctx_t*>(jpeg_ctx.src);
  source_ctx->base.init_source = init_source_cb;
  source_ctx->base.fill_input_buffer = fill_input_buffer_cb;
  source_ctx->base.skip_input_data = skip_input_data_cb;
  source_ctx->base.resync_to_restart = jpeg_resync_to_restart; // default
  source_ctx->base.term_source = term_source_cb;
  source_ctx->data = reinterpret_cast<const JOCTET*>(data);
  source_ctx->len = len;
  source_ctx->base.bytes_in_buffer = len;
  source_ctx->base.next_input_byte = source_ctx->data;

  jpeg_save_markers(&jpeg_ctx, APP1, 0xffff);
}

inline uint8_t clamped_cmyk_rgb_convert(uint8_t k, uint8_t cmy) {
  // Inspired from Pillow:
  // https://github.com/python-pillow/Pillow/blob/07623d1a7cc65206a5355fba2ae256550bfcaba6/src/libImaging/Convert.c#L568-L569
  int v = k * cmy + 128;
  v = ((v >> 8) + v) >> 8;
  return std::clamp(k - v, 0, 255);
}

void convert_line_cmyk_to_rgb(
    int width,
    const uint8_t* cmyk_line,
    uint8_t* rgb_line) {
  for (int i = 0; i < width; ++i) {
    int c = cmyk_line[i * 4 + 0];
    int m = cmyk_line[i * 4 + 1];
    int y = cmyk_line[i * 4 + 2];
    int k = cmyk_line[i * 4 + 3];

    rgb_line[i * 3 + 0] = clamped_cmyk_rgb_convert(k, 255 - c);
    rgb_line[i * 3 + 1] = clamped_cmyk_rgb_convert(k, 255 - m);
    rgb_line[i * 3 + 2] = clamped_cmyk_rgb_convert(k, 255 - y);
  }
}

inline uint8_t rgb_to_gray(int r, int g, int b) {
  // Inspired from Pillow:
  // https://github.com/python-pillow/Pillow/blob/07623d1a7cc65206a5355fba2ae256550bfcaba6/src/libImaging/Convert.c#L226
  return (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
}

void convert_line_cmyk_to_gray(
    int width,
    const uint8_t* cmyk_line,
    uint8_t* gray_line) {
  for (int i = 0; i < width; ++i) {
    int c = cmyk_line[i * 4 + 0];
    int m = cmyk_line[i * 4 + 1];
    int y = cmyk_line[i * 4 + 2];
    int k = cmyk_line[i * 4 + 3];

    int r = clamped_cmyk_rgb_convert(k, 255 - c);
    int g = clamped_cmyk_rgb_convert(k, 255 - m);
    int b = clamped_cmyk_rgb_convert(k, 255 - y);

    gray_line[i] = rgb_to_gray(r, g, b);
  }
}

int fetch_jpeg_exif_orientation(j_decompress_ptr jpeg_ctx) {
  STD_TORCH_CHECK(jpeg_ctx != nullptr, "jpeg_ctx cannot be null");

  // Check for Exif marker APP1
  jpeg_saved_marker_ptr exif_marker = 0;
  jpeg_saved_marker_ptr cmarker = jpeg_ctx->marker_list;
  while (cmarker && exif_marker == 0) {
    if (cmarker->marker == APP1) {
      exif_marker = cmarker;
    }
    cmarker = cmarker->next;
  }

  if (!exif_marker) {
    return -1;
  }

  constexpr size_t start_offset = 6;
  if (exif_marker->data_length <= start_offset) {
    return -1;
  }

  auto* exif_data_ptr = exif_marker->data + start_offset;
  auto size = exif_marker->data_length - start_offset;

  return fetch_exif_orientation(exif_data_ptr, size);
}

} // namespace

torch::stable::Tensor decode_jpeg(
    const torch::stable::Tensor& data,
    int64_t mode) {
  validate_encoded_data(data);

  // See error handling below why these are optional
  std::optional<torch::stable::Tensor> output;
  std::optional<torch::stable::Tensor> cmyk_line_tensor;

  auto datap = data.const_data_ptr<uint8_t>();

  jpeg_decompress_struct jpeg_ctx;

  // Setup error handling. libjpeg uses setjmp/longjmp for that. longjmp does
  // not unwind C++ stack frames, so destructors of objects created after setjmp
  // won't run. We use std::optional to declare tensors before setjmp while
  // deferring construction, and explicitly reset them on the error path.
  error_ctx_t error_ctx;
  jpeg_ctx.err = jpeg_std_error(&error_ctx.base);
  error_ctx.base.error_exit = error_exit_cb;
  // Establish the setjmp return context for error_exit_cb to use.
  if (setjmp(error_ctx.setjmp_buffer)) {
    // Release any tensors that may have been allocated after setjmp.
    cmyk_line_tensor.reset();
    output.reset();

    // If we get here, the JPEG code has signaled an error.
    // We need to clean up the JPEG object.
    jpeg_destroy_decompress(&jpeg_ctx);
    STD_TORCH_CHECK(false, error_ctx.last_error_message);
  }

  jpeg_create_decompress(&jpeg_ctx);
  set_source_ctx(jpeg_ctx, datap, data.numel());

  jpeg_read_header(&jpeg_ctx, TRUE);

  int num_output_channels = jpeg_ctx.num_components;
  bool cmyk_to_rgb_or_gray = false;

  // TODO_IMAGE wait, what does this return on a CMYK image when mode is
  // UNCHANGED?
  if (mode != kImageReadModeUnchanged) {
    // libjpeg can't convert CMYK/YCCK straight to gray or RGB, so for those we
    // decode as CMYK and convert the lines ourselves (see the scanline loop),
    // like:
    // https://github.com/tensorflow/tensorflow/blob/86871065265b04e0db8ca360c046421efb2bdeb4/tensorflow/core/lib/jpeg/jpeg_mem.cc#L284-L313
    cmyk_to_rgb_or_gray = jpeg_ctx.jpeg_color_space == JCS_CMYK ||
        jpeg_ctx.jpeg_color_space == JCS_YCCK;
    switch (mode) {
      case kImageReadModeGray:
        jpeg_ctx.out_color_space =
            cmyk_to_rgb_or_gray ? JCS_CMYK : JCS_GRAYSCALE;
        num_output_channels = 1;
        break;
      case kImageReadModeRgb:
        jpeg_ctx.out_color_space = cmyk_to_rgb_or_gray ? JCS_CMYK : JCS_RGB;
        num_output_channels = 3;
        break;
      default:
        jpeg_destroy_decompress(&jpeg_ctx);
        STD_TORCH_CHECK(
            false, "The provided mode is not supported for JPEG files");
    }

    jpeg_calc_output_dimensions(&jpeg_ctx);
  }

  jpeg_start_decompress(&jpeg_ctx);

  int height = jpeg_ctx.output_height;
  int width = jpeg_ctx.output_width;

  int stride = width * num_output_channels; // we want channel-last output
  output = torch::stable::empty(
      {int64_t(height), int64_t(width), num_output_channels}, kStableUInt8);
  auto outputp = output->mutable_data_ptr<uint8_t>();

  if (cmyk_to_rgb_or_gray) {
    cmyk_line_tensor = torch::stable::empty({int64_t(width), 4}, kStableUInt8);
  }

  while (jpeg_ctx.output_scanline < jpeg_ctx.output_height) {
    if (cmyk_to_rgb_or_gray) {
      auto cmyk_line_ptr = cmyk_line_tensor->mutable_data_ptr<uint8_t>();
      jpeg_read_scanlines(&jpeg_ctx, &cmyk_line_ptr, 1);

      if (num_output_channels == 3) {
        convert_line_cmyk_to_rgb(width, cmyk_line_ptr, outputp);
      } else if (num_output_channels == 1) {
        convert_line_cmyk_to_gray(width, cmyk_line_ptr, outputp);
      }
    } else {
      jpeg_read_scanlines(&jpeg_ctx, &outputp, 1);
    }
    outputp += stride;
  }

  // EXIF markers were parsed during jpeg_read_header so this is just an
  // in-memory lookup (i.e. we're not going back to the beginning of the file)
  int exif_orientation = fetch_jpeg_exif_orientation(&jpeg_ctx);

  jpeg_finish_decompress(&jpeg_ctx);
  jpeg_destroy_decompress(&jpeg_ctx);
  return exif_orientation_transform(
      stable_permute(*output, {2, 0, 1}), exif_orientation);
}

} // namespace facebook::torchcodec

#endif // !JPEG_FOUND
