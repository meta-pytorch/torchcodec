// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "EncodeJpeg.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !TORCHCODEC_ENABLE_JPEG

namespace facebook::torchcodec {

void encode_jpeg_to_io(
    [[maybe_unused]] const torch::stable::Tensor& img,
    [[maybe_unused]] int64_t quality,
    [[maybe_unused]] IOInterface& io) {
  STD_TORCH_CHECK(
      false,
      "encode_jpeg: torchcodec was not compiled with libjpeg support. "
      "Rebuild torchcodec in an environment where libjpeg-turbo (and its "
      "development headers) are available. If you see this error in a prebuilt "
      "wheel, please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

#include <jpeglib.h>
#include <setjmp.h>

#include <cstdint>
#include <exception>
#include <vector>

namespace facebook::torchcodec {

namespace {

constexpr int OUTPUT_BUFFER_SIZE = 64 * 1024;

// Error context passed to libjpeg. Same shape and rationale as the one in
// DecodeJpeg.cpp: the jpeg_error_mgr base must be the first field so we can
// cast the `err` pointer libjpeg hands back to our callbacks into an ErrorCtx*.
struct ErrorCtx {
  jpeg_error_mgr base;
  char last_error_message[JMSG_LENGTH_MAX];
  jmp_buf setjmp_buffer;
};

void error_exit_cb(j_common_ptr jpeg_ctx) {
  auto* error_ctx = reinterpret_cast<ErrorCtx*>(jpeg_ctx->err);
  error_ctx->base.format_message(jpeg_ctx, error_ctx->last_error_message);
  longjmp(error_ctx->setjmp_buffer, 1);
}

// Custom libjpeg destination manager that streams compressed bytes into an
// IOInterface. jpeg_destination_mgr must be the first field so libjpeg's
// `dest` pointer can be cast back to an IODestMgr* in the callbacks.
//
// The io.write() calls may throw (e.g. a Python file-like raising, or a disk
// error). We must not let a C++ exception unwind through libjpeg's C stack, so
// each callback catches it and stashes it in `write_error`; the caller rethrows
// it after compression finishes. On error we stop writing but let libjpeg run
// to completion (its output is discarded).
struct IODestMgr {
  jpeg_destination_mgr pub;
  IOInterface* io;
  std::vector<JOCTET> buffer;
  std::exception_ptr write_error;
};

void init_destination(j_compress_ptr jpeg_ctx) {
  auto* dest = reinterpret_cast<IODestMgr*>(jpeg_ctx->dest);
  dest->pub.next_output_byte = dest->buffer.data();
  dest->pub.free_in_buffer = dest->buffer.size();
}

boolean empty_output_buffer(j_compress_ptr jpeg_ctx) {
  auto* dest = reinterpret_cast<IODestMgr*>(jpeg_ctx->dest);
  if (!dest->write_error) {
    try {
      dest->io->write(
          dest->buffer.data(), static_cast<int>(dest->buffer.size()));
    } catch (...) {
      dest->write_error = std::current_exception();
    }
  }
  dest->pub.next_output_byte = dest->buffer.data();
  dest->pub.free_in_buffer = dest->buffer.size();
  return TRUE;
}

void term_destination(j_compress_ptr jpeg_ctx) {
  auto* dest = reinterpret_cast<IODestMgr*>(jpeg_ctx->dest);
  if (dest->write_error) {
    return;
  }
  int64_t remaining =
      static_cast<int64_t>(dest->buffer.size() - dest->pub.free_in_buffer);
  if (remaining > 0) {
    try {
      dest->io->write(dest->buffer.data(), static_cast<int>(remaining));
    } catch (...) {
      dest->write_error = std::current_exception();
    }
  }
}

void compress_jpeg(
    jpeg_compress_struct& jpeg_ctx,
    ErrorCtx& error_ctx,
    IODestMgr& dest,
    const uint8_t* input_ptr,
    int width,
    int height,
    int num_channels,
    int quality) {
  if (setjmp(error_ctx.setjmp_buffer)) {
    jpeg_destroy_compress(&jpeg_ctx);
    STD_TORCH_CHECK(false, error_ctx.last_error_message);
  }

  jpeg_create_compress(&jpeg_ctx);

  jpeg_ctx.image_width = static_cast<JDIMENSION>(width);
  jpeg_ctx.image_height = static_cast<JDIMENSION>(height);
  jpeg_ctx.input_components = num_channels;
  jpeg_ctx.in_color_space = num_channels == 1 ? JCS_GRAYSCALE : JCS_RGB;

  jpeg_set_defaults(&jpeg_ctx);
  jpeg_set_quality(&jpeg_ctx, quality, /*force_baseline=*/TRUE);

  dest.pub.init_destination = &init_destination;
  dest.pub.empty_output_buffer = &empty_output_buffer;
  dest.pub.term_destination = &term_destination;
  jpeg_ctx.dest = &dest.pub;

  jpeg_start_compress(&jpeg_ctx, /*write_all_tables=*/TRUE);

  const int64_t stride = static_cast<int64_t>(width) * num_channels;
  auto* row = const_cast<JSAMPROW>(input_ptr);
  while (jpeg_ctx.next_scanline < jpeg_ctx.image_height) {
    jpeg_write_scanlines(&jpeg_ctx, &row, /*num_lines=*/1);
    row += stride;
  }

  jpeg_finish_compress(&jpeg_ctx);
  jpeg_destroy_compress(&jpeg_ctx);
}

} // namespace

// Important: see Note [libjpeg error handling] in the jpeg decoder: everything
// applies here too. We must not throw a C++ exception through libjpeg's C stack
// (and callbacks), and we must not allocate anything that needs proper
// destruction in a function that defines a setjmp() point. The IODestMgr (which
// owns a std::vector) therefore lives here, outside compress_jpeg's setjmp.
void encode_jpeg_to_io(
    const torch::stable::Tensor& img,
    int64_t quality,
    IOInterface& io) {
  STD_TORCH_CHECK(
      img.device().type() == kStableCPU,
      "Input tensor must be on the CPU, got a tensor on ",
      device_type_name(img.device().type()),
      ".");
  STD_TORCH_CHECK(
      img.scalar_type() == kStableUInt8, "Input tensor dtype should be uint8");
  STD_TORCH_CHECK(
      img.dim() == 3, "Input data should be a 3-dimensional tensor");

  const auto num_channels = static_cast<int>(img.size(0));
  const auto height = static_cast<int>(img.size(1));
  const auto width = static_cast<int>(img.size(2));
  STD_TORCH_CHECK(
      num_channels == 1 || num_channels == 3,
      "The number of channels should be 1 or 3, got: ",
      num_channels);

  // libjpeg consumes samples channels-last (HWC), one contiguous row at a time.
  const auto input = torch::stable::contiguous(stable_permute(img, {1, 2, 0}));

  // Owned here rather than in compress_jpeg(): that function runs the
  // setjmp/longjmp dance and must keep these alive across it.
  jpeg_compress_struct jpeg_ctx;
  ErrorCtx error_ctx;
  jpeg_ctx.err = jpeg_std_error(&error_ctx.base);
  error_ctx.base.error_exit = error_exit_cb;

  IODestMgr dest;
  dest.io = &io;
  dest.buffer.resize(OUTPUT_BUFFER_SIZE);

  compress_jpeg(
      jpeg_ctx,
      error_ctx,
      dest,
      input.const_data_ptr<uint8_t>(),
      width,
      height,
      num_channels,
      static_cast<int>(quality));

  if (dest.write_error) {
    std::rethrow_exception(dest.write_error);
  }
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_JPEG
