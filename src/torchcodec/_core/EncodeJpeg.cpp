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

// The libjpeg compression pipeline. Owns the setjmp() point, so per Note
// [libjpeg error handling] in DecodeJpeg.cpp it must not declare anything with
// a non-trivial destructor, and everything it touches on the longjmp error path
// (cinfo, error_ctx, jpeg_buf) is owned by the caller and passed by reference:
// automatic locals modified between setjmp() and longjmp() have indeterminate
// values once we jump back into the setjmp block.
//
// libjpeg allocates jpeg_buf with malloc() and grows it as needed; ownership is
// handed back to the caller via the reference.
void compress_jpeg(
    jpeg_compress_struct& cinfo,
    ErrorCtx& error_ctx,
    const uint8_t* input_ptr,
    int width,
    int height,
    int channels,
    int quality,
    uint8_t*& jpeg_buf,
    JpegSizeType& jpeg_size) {
  if (setjmp(error_ctx.setjmp_buffer)) {
    jpeg_destroy_compress(&cinfo);
    std::free(jpeg_buf);
    jpeg_buf = nullptr;
    STD_TORCH_CHECK(false, error_ctx.last_error_message);
  }

  jpeg_create_compress(&cinfo);

  cinfo.image_width = static_cast<JDIMENSION>(width);
  cinfo.image_height = static_cast<JDIMENSION>(height);
  cinfo.input_components = channels;
  cinfo.in_color_space = channels == 1 ? JCS_GRAYSCALE : JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, /*force_baseline=*/TRUE);
  jpeg_mem_dest(&cinfo, &jpeg_buf, &jpeg_size);
  jpeg_start_compress(&cinfo, /*write_all_tables=*/TRUE);

  const int64_t stride = static_cast<int64_t>(width) * channels;
  // jpeg_write_scanlines wants a mutable JSAMPARRAY, but it only reads the
  // rows.
  auto* row = const_cast<JSAMPROW>(input_ptr);
  while (cinfo.next_scanline < cinfo.image_height) {
    jpeg_write_scanlines(&cinfo, &row, /*num_lines=*/1);
    row += stride;
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
}

} // namespace

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

  const auto channels = static_cast<int>(img.size(0));
  const auto height = static_cast<int>(img.size(1));
  const auto width = static_cast<int>(img.size(2));
  STD_TORCH_CHECK(
      channels == 1 || channels == 3,
      "The number of channels should be 1 or 3, got: ",
      channels);

  // libjpeg consumes samples channels-last (HWC), one contiguous row at a time.
  const auto input = torch::stable::contiguous(stable_permute(img, {1, 2, 0}));

  // Owned here rather than in compress_jpeg(): that function runs the
  // setjmp/longjmp dance and must keep these alive across it (see its doc).
  jpeg_compress_struct cinfo;
  ErrorCtx error_ctx;
  cinfo.err = jpeg_std_error(&error_ctx.base);
  error_ctx.base.error_exit = error_exit_cb;

  uint8_t* jpeg_buf = nullptr;
  JpegSizeType jpeg_size = 0;
  compress_jpeg(
      cinfo,
      error_ctx,
      input.const_data_ptr<uint8_t>(),
      width,
      height,
      channels,
      static_cast<int>(quality),
      jpeg_buf,
      jpeg_size);

  // Hand the malloc'd buffer to the tensor; from_blob's deleter frees it.
  return torch::stable::from_blob(
      jpeg_buf,
      {static_cast<int64_t>(jpeg_size)},
      {1},
      StableDevice(kStableCPU),
      kStableUInt8,
      [](void* ptr) { std::free(ptr); });
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_JPEG
