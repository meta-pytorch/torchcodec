// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "EncodePng.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !TORCHCODEC_ENABLE_PNG

namespace facebook::torchcodec {

void encode_png_to_io(
    [[maybe_unused]] const torch::stable::Tensor& img,
    [[maybe_unused]] int64_t compression_level,
    [[maybe_unused]] IOInterface& io) {
  STD_TORCH_CHECK(
      false,
      "encode_png: torchcodec was not compiled with libpng support. "
      "Rebuild torchcodec in an environment where libpng (and its development "
      "headers) are available. If you see this error in a prebuilt wheel, "
      "please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

#include <png.h>
#include <setjmp.h>

#include <cstdint>
#include <cstdio>
#include <exception>

namespace facebook::torchcodec {

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

// libpng calls write_callback with chunks of encoded output, which we forward
// to the IOInterface. The io.write() call may throw (e.g. a Python file-like
// raising, or a disk error). We must not let a C++ exception unwind through
// libpng's C stack, so we catch it and stash it in `write_error`; the caller
// rethrows it after encoding finishes. On error we stop writing but let libpng
// run to completion (its output is discarded).
struct IOWriteCtx {
  IOInterface* io;
  std::exception_ptr write_error;
};

void write_callback(png_structp png_ptr, png_bytep data, png_size_t length) {
  auto* ctx = static_cast<IOWriteCtx*>(png_get_io_ptr(png_ptr));
  if (ctx->write_error) {
    return;
  }
  try {
    ctx->io->write(
        reinterpret_cast<const uint8_t*>(data), static_cast<int>(length));
  } catch (...) {
    ctx->write_error = std::current_exception();
  }
}

void write_png_to_io(
    png_structp& png_write,
    png_infop& info_ptr,
    ErrorCtx& error_ctx,
    IOWriteCtx& write_ctx,
    const uint8_t* input_ptr,
    int64_t width,
    int64_t height,
    int64_t num_channels,
    int64_t compression_level) {
  if (setjmp(png_jmpbuf(png_write)) != 0) {
    png_destroy_write_struct(&png_write, &info_ptr);
    STD_TORCH_CHECK(false, "encode_png failed: ", error_ctx.error_message);
  }

  png_set_write_fn(png_write, &write_ctx, write_callback, /*flush_fn=*/nullptr);

  const int color_type =
      (num_channels == 1) ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB;
  png_set_IHDR(
      png_write,
      info_ptr,
      static_cast<png_uint_32>(width),
      static_cast<png_uint_32>(height),
      /*bit_depth=*/8,
      color_type,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT);
  png_set_compression_level(png_write, static_cast<int>(compression_level));
  png_write_info(png_write, info_ptr);

  const int64_t stride = width * num_channels;
  for (int64_t row = 0; row < height; ++row) {
    png_write_row(png_write, input_ptr + row * stride);
  }
  png_write_end(png_write, info_ptr);
}

} // namespace

// Important: see Note [libjpeg error handling] in the jpeg decoder: everything
// applies here too. We must not throw a C++ exception through libpng's C stack
// (and callbacks), and we must not allocate anything that needs proper
// destruction in a function that defines a setjmp() point. The IOWriteCtx
// (which owns a std::exception_ptr) therefore lives here, outside
// write_png_to_io's setjmp.
void encode_png_to_io(
    const torch::stable::Tensor& img,
    int64_t compression_level,
    IOInterface& io) {
  STD_TORCH_CHECK(
      compression_level >= 0 && compression_level <= 9,
      "Compression level should be between 0 and 9, got ",
      compression_level,
      ".");
  STD_TORCH_CHECK(
      img.device().type() == kStableCPU,
      "Input tensor must be on the CPU, got a tensor on ",
      device_type_name(img.device().type()),
      ".");
  STD_TORCH_CHECK(
      img.scalar_type() == kStableUInt8,
      "Input tensor must have uint8 data type, got ",
      torch::headeronly::toString(img.scalar_type()),
      ".");
  STD_TORCH_CHECK(
      img.dim() == 3,
      "Input tensor must be a 3-dimensional (C, H, W) tensor, got ",
      img.dim(),
      " dimensions.");

  const int64_t num_channels = img.size(0);
  const int64_t height = img.size(1);
  const int64_t width = img.size(2);
  STD_TORCH_CHECK(
      num_channels == 1 || num_channels == 3,
      "The number of channels should be 1 or 3, got ",
      num_channels,
      ".");

  // libpng writes rows in HWC interleaved order, so permute from CHW. This
  // contiguous copy is kept alive for the whole encoding below.
  auto input = torch::stable::contiguous(stable_permute(img, {1, 2, 0}));

  auto png_write =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  STD_TORCH_CHECK(
      png_write != nullptr, "libpng write structure allocation failed!");
  auto info_ptr = png_create_info_struct(png_write);
  if (info_ptr == nullptr) {
    png_destroy_write_struct(&png_write, nullptr);
    STD_TORCH_CHECK(false, "libpng info structure allocation failed!");
  }

  ErrorCtx error_ctx;
  png_set_error_fn(png_write, &error_ctx, error_callback, /*warn_fn=*/nullptr);

  IOWriteCtx write_ctx;
  write_ctx.io = &io;

  write_png_to_io(
      png_write,
      info_ptr,
      error_ctx,
      write_ctx,
      input.const_data_ptr<uint8_t>(),
      width,
      height,
      num_channels,
      compression_level);

  png_destroy_write_struct(&png_write, &info_ptr);

  if (write_ctx.write_error) {
    std::rethrow_exception(write_ctx.write_error);
  }
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_PNG
