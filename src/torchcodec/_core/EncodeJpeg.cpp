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

void encode_jpeg(
    [[maybe_unused]] const torch::stable::Tensor& img,
    [[maybe_unused]] int64_t quality,
    [[maybe_unused]] IOInterface& interface) {
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
// IOInterface. The jpeg_destination_mgr must be the first field so libjpeg's
// `dest` pointer can be cast back to an IOCtx* in the callbacks.
struct IOCtx {
  jpeg_destination_mgr jpeg_dest_mgr;
  IOInterface* interface;
  // libjpeg writes to this buffer in chunks, and we flush it to the IOInterface
  // whenever it fills up.
  std::vector<JOCTET> buffer;
  std::exception_ptr interface_exception;
};

// Flushes the first `size` bytes of io_ctx->buffer to the IOInterface. The
// interface->write() call may throw (e.g. a Python file-like raising, or a disk
// error), but a C++ exception must not unwind through libjpeg's C stack. So we
// capture it and abort compression by longjmp-ing to the setjmp point in
// compress_jpeg, which rethrows it.
void write_or_longjmp(j_compress_ptr jpeg_ctx, IOCtx* io_ctx, int size) {
  try {
    io_ctx->interface->write(io_ctx->buffer.data(), size);
    return;
  } catch (...) {
    io_ctx->interface_exception = std::current_exception();
  }
  longjmp(reinterpret_cast<ErrorCtx*>(jpeg_ctx->err)->setjmp_buffer, 1);
}

// Called by jpeg_start_compress() before any data is written.
void init_destination(j_compress_ptr jpeg_ctx) {
  auto* io_ctx = reinterpret_cast<IOCtx*>(jpeg_ctx->dest);
  io_ctx->jpeg_dest_mgr.next_output_byte = io_ctx->buffer.data();
  io_ctx->jpeg_dest_mgr.free_in_buffer = io_ctx->buffer.size();
}

// Called whenever libjpeg runs out of space in the output buffer. We flush the
// buffer to the IOInterface and reset the buffer.
boolean empty_output_buffer(j_compress_ptr jpeg_ctx) {
  auto* io_ctx = reinterpret_cast<IOCtx*>(jpeg_ctx->dest);
  write_or_longjmp(jpeg_ctx, io_ctx, static_cast<int>(io_ctx->buffer.size()));
  io_ctx->jpeg_dest_mgr.next_output_byte = io_ctx->buffer.data();
  io_ctx->jpeg_dest_mgr.free_in_buffer = io_ctx->buffer.size();
  return TRUE;
}

// called by jpeg_finish_compress() after all data has been written. There might
// still be some data in the buffer which needs to be flushed to the
// IOInterface.
void term_destination(j_compress_ptr jpeg_ctx) {
  auto* io_ctx = reinterpret_cast<IOCtx*>(jpeg_ctx->dest);
  int size = static_cast<int>(
      io_ctx->buffer.size() - io_ctx->jpeg_dest_mgr.free_in_buffer);
  if (size > 0) {
    write_or_longjmp(jpeg_ctx, io_ctx, size);
  }
}

void compress_jpeg(
    jpeg_compress_struct& jpeg_ctx,
    ErrorCtx& error_ctx,
    IOCtx& io_ctx,
    const uint8_t* input_ptr,
    int width,
    int height,
    int num_channels,
    int quality) {
  if (setjmp(error_ctx.setjmp_buffer)) {
    jpeg_destroy_compress(&jpeg_ctx);
    // We land here on either a libjpeg error (via error_exit_cb) or a failed
    // write (via write_or_longjmp). The latter sets interface_exception, which
    // we rethrow to surface the original exception.
    if (io_ctx.interface_exception) {
      std::rethrow_exception(io_ctx.interface_exception);
    } else {
      STD_TORCH_CHECK(false, error_ctx.last_error_message);
    }
  }

  jpeg_create_compress(&jpeg_ctx);

  jpeg_ctx.image_width = static_cast<JDIMENSION>(width);
  jpeg_ctx.image_height = static_cast<JDIMENSION>(height);
  jpeg_ctx.input_components = num_channels;
  jpeg_ctx.in_color_space = num_channels == 1 ? JCS_GRAYSCALE : JCS_RGB;

  jpeg_set_defaults(&jpeg_ctx);
  jpeg_set_quality(&jpeg_ctx, quality, /*force_baseline=*/TRUE);

  io_ctx.jpeg_dest_mgr.init_destination = &init_destination;
  io_ctx.jpeg_dest_mgr.empty_output_buffer = &empty_output_buffer;
  io_ctx.jpeg_dest_mgr.term_destination = &term_destination;
  jpeg_ctx.dest = &io_ctx.jpeg_dest_mgr;

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
// destruction in a function that defines a setjmp() point. The IOCtx (which
// owns a std::vector) therefore lives here, outside compress_jpeg's setjmp.
void encode_jpeg(
    const torch::stable::Tensor& img,
    int64_t quality,
    IOInterface& interface) {
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

  IOCtx io_ctx;
  io_ctx.interface = &interface;
  io_ctx.buffer.resize(OUTPUT_BUFFER_SIZE);

  compress_jpeg(
      jpeg_ctx,
      error_ctx,
      io_ctx,
      input.const_data_ptr<uint8_t>(),
      width,
      height,
      num_channels,
      static_cast<int>(quality));
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_JPEG
