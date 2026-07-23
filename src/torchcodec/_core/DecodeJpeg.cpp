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

#if !TORCHCODEC_ENABLE_JPEG

namespace facebook::torchcodec {

torch::stable::Tensor decode_jpeg(
    [[maybe_unused]] const torch::stable::Tensor& data,
    [[maybe_unused]] int64_t mode) {
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
#include <tuple>

#include "Exif.h"
#include "ImageCommon.h"

namespace facebook::torchcodec {

using namespace exif_private;

namespace {

// Main libjpeg docs:
// https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/doc/libjpeg.txt

constexpr uint16_t EXIF_APP1 = 0xe1;

// Helpers to convert a CMYK line to RGB or grayscale, since libjpeg doesn't
// handle that directly.
using CmykLineConverterFn = void (*)(int, const uint8_t*, uint8_t*);

struct CMYKHelper {
  uint8_t* cmyk_line_ptr = nullptr;
  // This is either convert_line_cmyk_to_rgb or convert_line_cmyk_to_gray
  CmykLineConverterFn convert_fn = nullptr;
};

inline uint8_t cmyk_to_rgb(uint8_t k, uint8_t cmy) {
  // Inspired from Pillow:
  // https://github.com/python-pillow/Pillow/blob/07623d1a7cc65206a5355fba2ae256550bfcaba6/src/libImaging/Convert.c#L568-L569
  int v = k * cmy + 128;
  v = ((v >> 8) + v) >> 8;
  return std::clamp(k - v, 0, 255);
}

void convert_line_cmyk_to_rgb(
    int width,
    const uint8_t* cmyk_line_ptr,
    uint8_t* output_ptr) {
  for (int i = 0; i < width; ++i) {
    int c = cmyk_line_ptr[i * 4 + 0];
    int m = cmyk_line_ptr[i * 4 + 1];
    int y = cmyk_line_ptr[i * 4 + 2];
    int k = cmyk_line_ptr[i * 4 + 3];

    output_ptr[i * 3 + 0] = cmyk_to_rgb(k, 255 - c);
    output_ptr[i * 3 + 1] = cmyk_to_rgb(k, 255 - m);
    output_ptr[i * 3 + 2] = cmyk_to_rgb(k, 255 - y);
  }
}

inline uint8_t rgb_to_gray(int r, int g, int b) {
  // Inspired from Pillow:
  // https://github.com/python-pillow/Pillow/blob/07623d1a7cc65206a5355fba2ae256550bfcaba6/src/libImaging/Convert.c#L226
  return (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
}

void convert_line_cmyk_to_gray(
    int width,
    const uint8_t* cmyk_line_ptr,
    uint8_t* output_ptr) {
  for (int i = 0; i < width; ++i) {
    int c = cmyk_line_ptr[i * 4 + 0];
    int m = cmyk_line_ptr[i * 4 + 1];
    int y = cmyk_line_ptr[i * 4 + 2];
    int k = cmyk_line_ptr[i * 4 + 3];

    int r = cmyk_to_rgb(k, 255 - c);
    int g = cmyk_to_rgb(k, 255 - m);
    int b = cmyk_to_rgb(k, 255 - y);

    output_ptr[i] = rgb_to_gray(r, g, b);
  }
}

ExifOrientation fetch_jpeg_exif_orientation(j_decompress_ptr jpeg_ctx) {
  STD_TORCH_CHECK(jpeg_ctx != nullptr, "jpeg_ctx cannot be null");

  jpeg_saved_marker_ptr exif_marker = jpeg_ctx->marker_list;
  while (exif_marker != nullptr) {
    if (exif_marker->marker == EXIF_APP1) {
      break;
    }
    exif_marker = exif_marker->next;
  }

  if (exif_marker == nullptr) {
    return ExifOrientation::Unspecified;
  }

  constexpr size_t start_offset = 6;
  if (exif_marker->data_length <= start_offset) {
    return ExifOrientation::Unspecified;
  }

  auto* exif_data_ptr = exif_marker->data + start_offset;
  auto size = exif_marker->data_length - start_offset;

  return fetch_exif_orientation(exif_data_ptr, size);
}

// Error context that gets passed by libjpeg to its callbacks
// (error_exit_cb, fill_input_buffer_cb, etc.).
// libjpeg doesn't actually pass this ErrorCtx, it passes the jpeg_ctx object
// which has an `err` field. This `err` field is *still* not an ErrorCtx
// object, it's the jpeg_error_mgr base field, but we can cast it back to
// ErrorCtx in the callbacks because the struct and its first field share the
// same address.
struct ErrorCtx {
  jpeg_error_mgr base;
  char last_error_message[JMSG_LENGTH_MAX];
  jmp_buf setjmp_buffer;
};

// Callback called by libjpeg when an error occurs.
void error_exit_cb(j_common_ptr jpeg_ctx) {
  auto error_ctx = reinterpret_cast<ErrorCtx*>(jpeg_ctx->err);
  error_ctx->base.format_message(jpeg_ctx, error_ctx->last_error_message);
  longjmp(error_ctx->setjmp_buffer, 1);
}

// Callback called by libjpeg whenever it runs out of input data. We treat this
// as an error.
// Should we ever want to support truncated JPEGs, we could instead return a
// fake EOI.
boolean fill_input_buffer_cb(j_decompress_ptr jpeg_ctx) {
  auto error_ctx = reinterpret_cast<ErrorCtx*>(jpeg_ctx->err);
  strcpy(error_ctx->last_error_message, "Image is incomplete or truncated.");
  longjmp(error_ctx->setjmp_buffer, 1);
  return TRUE; // never reached, but keeps compiler happy
}

// Callback called when libjpeg wants to skip num_bytes worth of data.
void skip_input_data_cb(j_decompress_ptr jpeg_ctx, long num_bytes) {
  if (num_bytes <= 0) {
    return; // libjpeg docs say to ignore non-positive values
  }
  if (jpeg_ctx->src->bytes_in_buffer < static_cast<size_t>(num_bytes)) {
    // libjpeg requested to skip more data than is available. This path isn't
    // exercized in our tests.
    // In TorchVision this would return a fake EOI, but that's inconsistent with
    // our fill_input_buffer_cb, which treats truncated JPEGs as an error.
    // So we error here too.
    auto error_ctx = reinterpret_cast<ErrorCtx*>(jpeg_ctx->err);
    strcpy(
        error_ctx->last_error_message,
        "Skipped over more data than is available in the input buffer.");
    longjmp(error_ctx->setjmp_buffer, 1);
  } else {
    jpeg_ctx->src->next_input_byte += num_bytes;
    jpeg_ctx->src->bytes_in_buffer -= num_bytes;
  }
}

void init_source_cb(j_decompress_ptr) {}

void term_source_cb(j_decompress_ptr) {}

// Returns {num_output_channels, cmyk_to_rgb_or_gray}.
// jpeg_ctx.output_height and jpeg_ctx.output_width are available after this
// function returns.
std::tuple<int, bool> read_header_and_start(
    jpeg_decompress_struct& jpeg_ctx,
    ErrorCtx& error_ctx,
    const uint8_t* input_ptr,
    const size_t input_len,
    ImageReadMode mode) {
  if (setjmp(error_ctx.setjmp_buffer)) {
    // See Note [libjpeg error handling]
    jpeg_destroy_decompress(&jpeg_ctx);
    STD_TORCH_CHECK(false, error_ctx.last_error_message);
  }

  jpeg_create_decompress(&jpeg_ctx);

  jpeg_ctx.src = static_cast<jpeg_source_mgr*>(jpeg_ctx.mem->alloc_small(
      reinterpret_cast<j_common_ptr>(&jpeg_ctx),
      JPOOL_PERMANENT,
      sizeof(jpeg_source_mgr)));
  jpeg_ctx.src->init_source = init_source_cb;
  jpeg_ctx.src->fill_input_buffer = fill_input_buffer_cb;
  jpeg_ctx.src->skip_input_data = skip_input_data_cb;
  jpeg_ctx.src->resync_to_restart = jpeg_resync_to_restart; // default
  jpeg_ctx.src->term_source = term_source_cb;
  jpeg_ctx.src->bytes_in_buffer = input_len;
  jpeg_ctx.src->next_input_byte = input_ptr;

  // Tells libjpeg to save APP1 markers (EXIF) in memory for later retrieval.
  jpeg_save_markers(&jpeg_ctx, EXIF_APP1, 0xffff);

  jpeg_read_header(&jpeg_ctx, TRUE);

  // libjpeg natively decodes to UNCHANGED, GRAY or RGB. The alpha modes
  // (GRAY_ALPHA, RGB_ALPHA) are emulated in Python at the single
  // _decode_with_mode() callsite (see _image_decoders.py), which requests a
  // native mode here and appends an alpha channel. The cpp decode_jpeg is not a
  // public API and is only ever called from there, so the default branch is
  // unreachable.
  int num_output_channels = -1;
  switch (mode) {
    case ImageReadMode::UNCHANGED:
      num_output_channels = jpeg_ctx.num_components;
      break;
    case ImageReadMode::GRAY:
      num_output_channels = 1;
      break;
    case ImageReadMode::RGB:
      num_output_channels = 3;
      break;
    default:
      jpeg_destroy_decompress(&jpeg_ctx);
      STD_TORCH_CHECK(
          false,
          "Reached an unexpected code path while decoding a JPEG file to mode ",
          static_cast<int64_t>(mode),
          ". This should never happen, please report a bug to the TorchCodec repo.");
  }

  // libjpeg can't convert CMYK/YCCK straight to gray or RGB, so for those modes
  // we keep libjpeg's JCS_CMYK default as the output color-space, and convert
  // the lines ourselves (see the decode_rows loop). Similar to:
  // https://github.com/tensorflow/tensorflow/blob/86871065265b04e0db8ca360c046421efb2bdeb4/tensorflow/core/lib/jpeg/jpeg_mem.cc#L284-L313
  bool cmyk_to_rgb_or_gray = (jpeg_ctx.jpeg_color_space == JCS_CMYK ||
                              jpeg_ctx.jpeg_color_space == JCS_YCCK) &&
      (mode == ImageReadMode::GRAY || mode == ImageReadMode::RGB);

  // For other sources, ask libjpeg to convert to the requested color space.
  if (mode != ImageReadMode::UNCHANGED && !cmyk_to_rgb_or_gray) {
    if (mode == ImageReadMode::GRAY) {
      jpeg_ctx.out_color_space = JCS_GRAYSCALE;
    } else {
      STD_TORCH_CHECK(mode == ImageReadMode::RGB, "Should never reach here.");
      jpeg_ctx.out_color_space = JCS_RGB;
    }
  }
  jpeg_start_decompress(&jpeg_ctx);
  return {num_output_channels, cmyk_to_rgb_or_gray};
}

// The actual decoding loop, row by row.
void decode_rows(
    jpeg_decompress_struct& jpeg_ctx,
    ErrorCtx& error_ctx,
    uint8_t* output_ptr,
    int64_t stride,
    CMYKHelper& cmyk_helper) {
  if (setjmp(error_ctx.setjmp_buffer)) {
    // See Note [libjpeg error handling]
    jpeg_destroy_decompress(&jpeg_ctx);
    STD_TORCH_CHECK(false, error_ctx.last_error_message);
  }

  while (jpeg_ctx.output_scanline < jpeg_ctx.output_height) {
    if (cmyk_helper.cmyk_line_ptr != nullptr &&
        cmyk_helper.convert_fn != nullptr) {
      jpeg_read_scanlines(
          &jpeg_ctx, &cmyk_helper.cmyk_line_ptr, /*max_lines=*/1);
      cmyk_helper.convert_fn(
          jpeg_ctx.output_width, cmyk_helper.cmyk_line_ptr, output_ptr);
    } else {
      jpeg_read_scanlines(&jpeg_ctx, &output_ptr, /*max_lines=*/1);
    }
    output_ptr += stride;
  }
}

} // namespace

/* clang-format off */
//
// Note [libjpeg error handling]
//
// The structure of our code is:
//
// ```
// decode_jpeg():
//    torch::stable Tensor output;
//    read_header_and_start():
//        setjmp() {STD_TORCH_CHECK(false)}
//        libjpeg call stack that can trigger a callback, where we longjmp back
//          to the setjmp() above
//    decode_rows():
//        setjmp() {STD_TORCH_CHECK(false)}
//        libjpeg call stack that can trigger a callback, where we longjmp back
//          to the setjmp() above
// ```
//
// There's a reason for this structure: it is important that the `output` tensor
// is declared in a function that DOES NOT define a setjmp() (same for any other
// object that needs a non-trivial destructor). We previously had both the
// setjmp() and the tensor declaration within decode_jpeg(), with the
// read_header_and_start() and decode_rows() inlined there, and we would get a
// segault in the test_truncated_jpeg_raises() test:
//
// ```
// decode_jpeg():  (bad, UB territory)
//    torch::stable Tensor output;
//    setjmp() {STD_TORCH_CHECK(false)}
//    libjpeg call stack that can trigger a callback, where we longjmp back
//      to the setjmp() above
// ```
//
// Let's [try to] explain why. First, a bit of background on setjmp and longjmp:
// they're the closest that C can get to exceptions. You define a setjmp() point
// where you handle any error, and you get to that setjmp block by calling
// longjmp(). It's a glorified GOTO that restores [part of] the stack to the
// point of the setjmp() call:
//
// ```C
// jmp_buf setjmp_buffer;
// if (setjmp(setjmp_buffer) != 0) {
//    error path: exit gracefully, handle error, etc.
//  }
//
// // Some code down the line:
// longjmp(setjmp_buffer, /*err=*/ 1);  <-- jumps back to the setjmp point
// ```
//
// The setjmp_buffer works such that you don't enter the setjmp block on the
// first pass, but you do enter it when you longjmp back to it.
//
// libjpeg recommends using setjmp/longjmp but doesn't *force* us to: libjpeg
// just triggers callbacks on errors, like error_exit_cb(), and it's up to us to
// handle the error there. But we do use setjmp/longjmp because we can't
// directly throw exceptions within the callback itself (more on that later)
//
// Now, there's an important rule (C11 7.13.2.1):
// > the values of objects of automatic storage duration that are local to the
//   function containing the invocation of the corresponding setjmp macro that do
//   not have volatile-qualified type and have been changed between the setjmp
//   invocation and longjmp call are indeterminate.
//
// In other words, if we do what we had before:
//
// ```
// decode_jpeg():  (bad, UB territory)
//    torch::stable Tensor output;  // not a volatile
//
//    // setjump() in the same function as the tensor declaration
//    setjmp() {STD_TORCH_CHECK(false)}
//
//    modify output in the call stack
//
//    libjpeg call stack that can trigger a callback, where we longjmp back
//      to the setjmp() above
// ```
//
// then we have the `output` tensor which is:
// - not a volatile
// - in the same function that defines the setjmp()
// - modified
//
// which means `output` has indeterminate value within the  setjmp() block:
// bummer, that's where its destructor is needed, and that's why we segfault.
// You won't always segfault BTW, it depends on the compiler optimization level
// and other factors, but it's still UB.
//
// So, from there we have a few solutions:
//
// 1. Make the `output` tensor a volatile. This works, but making the
//    torch::stable::Tensor volatile isn't directly possible. We have to have a
//    volatile `torch::stable::Tensor*` pointer, and we lose RAII semantics.
//    Meh.
// 2. Not use setjmp/longjmp and just throw STD_TORCH_CHECK() directly in the
//    callbacks, i.e. `void error_exit_cb(...) { STD_TORCH_CHECK(false, "error"); }`.
//    The issue here is that this callback is invoked from libjpeg C code. And
//    for the exception to propagate up the stack, where the user may want to
//    catch it within a try block, the stack must 'unwind' the exception through
//    C code.  Whether C code can unwind C++ exceptions depends on how that C
//    code was compiled (there are compiler-specific flags). Usually, it does,
//    but we have no control over how libjpeg was built. It's just easier and
//    safer not to assume anything.
// 3. Don't declare `output` in the same function as the setjmp(). That's what
//    we do. That's why `decode_jpeg()` declares output and then calls
//    read_header_and_start() and decode_rows() where the setjmp points are
//    defined. Critically, it means that neither `read_header_and_start()` nor
//    `decore_rows()` should be declaring anything that needs proper destruction
//    within the setjmp/longjmp context.
//
/* clang-format on */

// TODO_IMAGE: align names. everywhere. this is input, other places it's data,
// other places it's something else. Should align here, in the header, in the
// custom op definition, etc. Across codecs.
torch::stable::Tensor decode_jpeg(
    const torch::stable::Tensor& input,
    int64_t mode) {
  validate_encoded_data(input);

  torch::stable::Tensor output;
  torch::stable::Tensor cmyk_line_tensor;

  jpeg_decompress_struct jpeg_ctx;
  ErrorCtx error_ctx;
  jpeg_ctx.err = jpeg_std_error(&error_ctx.base);
  error_ctx.base.error_exit = error_exit_cb;

  auto [num_output_channels, cmyk_to_rgb_or_gray] = read_header_and_start(
      jpeg_ctx,
      error_ctx,
      input.const_data_ptr<uint8_t>(),
      input.numel(),
      static_cast<ImageReadMode>(mode));

  // We want output to be channels last
  int64_t stride =
      static_cast<int64_t>(jpeg_ctx.output_width) * num_output_channels;
  output = torch::stable::empty(
      {int64_t(jpeg_ctx.output_height),
       int64_t(jpeg_ctx.output_width),
       num_output_channels},
      kStableUInt8);

  auto output_ptr = output.mutable_data_ptr<uint8_t>();

  CMYKHelper cmyk_helper = {nullptr, nullptr};
  if (cmyk_to_rgb_or_gray) {
    cmyk_line_tensor =
        torch::stable::empty({int64_t(jpeg_ctx.output_width), 4}, kStableUInt8);
    cmyk_helper.cmyk_line_ptr = cmyk_line_tensor.mutable_data_ptr<uint8_t>();
    if (num_output_channels == 3) {
      cmyk_helper.convert_fn = convert_line_cmyk_to_rgb;
    } else if (num_output_channels == 1) {
      cmyk_helper.convert_fn = convert_line_cmyk_to_gray;
    } else {
      STD_TORCH_CHECK(
          false,
          "Should never reach here, this is a bug in TorchCodec, please report");
    }
  }

  decode_rows(jpeg_ctx, error_ctx, output_ptr, stride, cmyk_helper);

  // EXIF markers were parsed during jpeg_read_header so this is just an
  // in-memory lookup (i.e. we're not going back to the beginning of the file)
  ExifOrientation exif_orientation = fetch_jpeg_exif_orientation(&jpeg_ctx);

  jpeg_finish_decompress(&jpeg_ctx);
  jpeg_destroy_decompress(&jpeg_ctx);
  return exif_orientation_transform(
      stable_permute(output, {2, 0, 1}), exif_orientation);
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_JPEG
