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

torch::stable::Tensor encode_jpeg(
    [[maybe_unused]] const torch::stable::Tensor& img,
    [[maybe_unused]] int64_t quality) {
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
#include <cstdlib>

namespace facebook::torchcodec {

namespace {

// For libjpeg <= 9b the out_size parameter of jpeg_mem_dest() is declared as
// `unsigned long`; later versions declare it as `size_t`.
#if !defined(JPEG_LIB_VERSION_MAJOR) || JPEG_LIB_VERSION_MAJOR < 9 || \
    (JPEG_LIB_VERSION_MAJOR == 9 && JPEG_LIB_VERSION_MINOR <= 2)
using JpegSizeType = unsigned long;
#else
using JpegSizeType = std::size_t;
#endif

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

int64_t compress_jpeg(
    jpeg_compress_struct& jpeg_ctx,
    ErrorCtx& error_ctx,
    const uint8_t* input_ptr,
    int width,
    int height,
    int num_channels,
    int quality,
    uint8_t*& jpeg_buf /* OUT */) {
  if (setjmp(error_ctx.setjmp_buffer)) {
    jpeg_destroy_compress(&jpeg_ctx);
    std::free(jpeg_buf);
    jpeg_buf = nullptr;
    STD_TORCH_CHECK(false, error_ctx.last_error_message);
  }

  jpeg_create_compress(&jpeg_ctx);

  jpeg_ctx.image_width = static_cast<JDIMENSION>(width);
  jpeg_ctx.image_height = static_cast<JDIMENSION>(height);
  jpeg_ctx.input_components = num_channels;
  jpeg_ctx.in_color_space = num_channels == 1 ? JCS_GRAYSCALE : JCS_RGB;

  jpeg_set_defaults(&jpeg_ctx);
  jpeg_set_quality(&jpeg_ctx, quality, /*force_baseline=*/TRUE);

  // libjpeg allocates jpeg_buf and grows it as needed
  JpegSizeType jpeg_size = 0;
  jpeg_mem_dest(&jpeg_ctx, &jpeg_buf, &jpeg_size);
  jpeg_start_compress(&jpeg_ctx, /*write_all_tables=*/TRUE);

  const int64_t stride = static_cast<int64_t>(width) * num_channels;
  auto* row = const_cast<JSAMPROW>(input_ptr);
  while (jpeg_ctx.next_scanline < jpeg_ctx.image_height) {
    jpeg_write_scanlines(&jpeg_ctx, &row, /*num_lines=*/1);
    row += stride;
  }

  jpeg_finish_compress(&jpeg_ctx);
  jpeg_destroy_compress(&jpeg_ctx);
  return static_cast<int64_t>(jpeg_size);
}

} // namespace

// Important: see Note [libjpeg error handling] in the jpeg decoder: everything
// applies here too. We must not throw a C++ exception through libjpeg's C stack
// (and callbacks), and we must not allocate anything that needs proper
// destruction in a function that defines a setjmp() point.
torch::stable::Tensor encode_jpeg(
    const torch::stable::Tensor& img,
    int64_t quality) {
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
  // setjmp/longjmp dance and must keep these alive across it
  jpeg_compress_struct jpeg_ctx;
  ErrorCtx error_ctx;
  jpeg_ctx.err = jpeg_std_error(&error_ctx.base);
  error_ctx.base.error_exit = error_exit_cb;

  uint8_t* jpeg_buf = nullptr;
  const int64_t jpeg_size = compress_jpeg(
      jpeg_ctx,
      error_ctx,
      input.const_data_ptr<uint8_t>(),
      width,
      height,
      num_channels,
      static_cast<int>(quality),
      jpeg_buf);

  return torch::stable::from_blob(
      jpeg_buf,
      {jpeg_size},
      {1},
      StableDevice(kStableCPU),
      kStableUInt8,
      [](void* ptr) { std::free(ptr); });
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_JPEG
