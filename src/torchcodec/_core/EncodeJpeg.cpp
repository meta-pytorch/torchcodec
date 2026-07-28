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
#include <cstring>

namespace facebook::torchcodec {

namespace {

// For libjpeg version <= 9b, the out_size parameter of jpeg_mem_dest() is
// declared as `unsigned long`; later versions declare it as `size_t`.
#if !defined(JPEG_LIB_VERSION_MAJOR) || JPEG_LIB_VERSION_MAJOR < 9 || \
    (JPEG_LIB_VERSION_MAJOR == 9 && JPEG_LIB_VERSION_MINOR <= 2)
using JpegSizeType = unsigned long;
#else
using JpegSizeType = size_t;
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
  auto error_ctx = reinterpret_cast<ErrorCtx*>(jpeg_ctx->err);
  error_ctx->base.format_message(jpeg_ctx, error_ctx->last_error_message);
  longjmp(error_ctx->setjmp_buffer, 1);
}

// Runs the whole libjpeg compression pipeline and returns the malloc'd output
// buffer via jpeg_buf_out / jpeg_size_out.
//
// This owns the setjmp() point. See Note [libjpeg error handling] in
// DecodeJpeg.cpp: nothing with a non-trivial destructor may be declared here,
// so the input and output tensors live in encode_jpeg() instead, and we only
// pass raw pointers around.
void encode_to_buffer(
    uint8_t* input_ptr,
    int width,
    int height,
    int channels,
    int64_t quality,
    uint8_t** jpeg_buf_out,
    JpegSizeType* jpeg_size_out) {
  jpeg_compress_struct cinfo;
  ErrorCtx error_ctx;
  cinfo.err = jpeg_std_error(&error_ctx.base);
  error_ctx.base.error_exit = error_exit_cb;

  if (setjmp(error_ctx.setjmp_buffer)) {
    jpeg_destroy_compress(&cinfo);
    if (*jpeg_buf_out != nullptr) {
      free(*jpeg_buf_out);
      *jpeg_buf_out = nullptr;
    }
    STD_TORCH_CHECK(false, error_ctx.last_error_message);
  }

  jpeg_create_compress(&cinfo);

  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = channels;
  cinfo.in_color_space = channels == 1 ? JCS_GRAYSCALE : JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);

  // libjpeg allocates the output buffer (with malloc) and grows it as needed.
  jpeg_mem_dest(&cinfo, jpeg_buf_out, jpeg_size_out);

  jpeg_start_compress(&cinfo, TRUE);

  int64_t stride = static_cast<int64_t>(width) * channels;
  JSAMPROW row = input_ptr;
  while (cinfo.next_scanline < cinfo.image_height) {
    jpeg_write_scanlines(&cinfo, &row, 1);
    row += stride;
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
}

} // namespace

torch::stable::Tensor encode_jpeg(
    [[maybe_unused]] const torch::stable::Tensor& img,
    [[maybe_unused]] int64_t quality) {
  STD_TORCH_CHECK(
      img.device().type() == kStableCPU,
      "Input tensor must be on the CPU, got a tensor on ",
      device_type_name(img.device().type()),
      ".");
  STD_TORCH_CHECK(
      img.scalar_type() == kStableUInt8, "Input tensor dtype should be uint8");
  STD_TORCH_CHECK(
      img.dim() == 3, "Input data should be a 3-dimensional tensor");

  int channels = img.size(0);
  int height = img.size(1);
  int width = img.size(2);
  STD_TORCH_CHECK(
      channels == 1 || channels == 3,
      "The number of channels should be 1 or 3, got: ",
      channels);

  // libjpeg wants the samples channels-last (HWC), one contiguous row at a
  // time.
  auto input = torch::stable::contiguous(stable_permute(img, {1, 2, 0}));
  auto input_ptr = input.mutable_data_ptr<uint8_t>();

  uint8_t* jpeg_buf = nullptr;
  JpegSizeType jpeg_size = 0;
  encode_to_buffer(
      input_ptr, width, height, channels, quality, &jpeg_buf, &jpeg_size);

  // Hand the malloc'd buffer to the output tensor; it frees it on destruction.
  auto deleter = [jpeg_buf](void*) { free(jpeg_buf); };
  return torch::stable::from_blob(
      jpeg_buf,
      {static_cast<int64_t>(jpeg_size)},
      {1},
      StableDevice(kStableCPU),
      kStableUInt8,
      deleter);
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_JPEG
