// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecodeWebp.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !WEBP_FOUND

namespace facebook::torchcodec {

torch::stable::Tensor decode_webp(
    const torch::stable::Tensor& data,
    int64_t mode) {
  STD_TORCH_CHECK(
      false,
      "decode_webp: torchcodec was not compiled with libwebp support. "
      "Rebuild torchcodec in an environment where libwebp (and its development "
      "headers) are available. If you see this error in a prebuilt wheel, "
      "please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

#include "webp/decode.h"
#include "webp/types.h"

#include "ImageCommon.h"

namespace facebook::torchcodec {

namespace {

// libwebp natively decodes to RGB or RGBA only. The grayscale modes are
// emulated in Python (see _image_decoders.py), so they never reach this op
// through the public decode_webp() API. Like decode_jpeg, we still reject them
// here for the benefit of anyone calling the raw op directly. This returns
// whether decode_webp should produce a 3-channel RGB tensor (true) or a
// 4-channel RGBA tensor (false).
bool should_return_rgb(ImageReadMode mode, bool has_alpha) {
  switch (mode) {
    case ImageReadMode::RGB:
      return true;
    case ImageReadMode::RGB_ALPHA:
      return false;
    case ImageReadMode::UNCHANGED:
      return !has_alpha;
    default:
      STD_TORCH_CHECK(
          false, "The provided mode is not supported for WebP files");
  }
}

} // namespace

torch::stable::Tensor decode_webp(
    const torch::stable::Tensor& input,
    int64_t mode) {
  validate_encoded_data(input);

  auto input_ptr = input.const_data_ptr<uint8_t>();
  auto input_size = input.numel();

  WebPBitstreamFeatures features;
  auto res = WebPGetFeatures(input_ptr, input_size, &features);
  STD_TORCH_CHECK(
      res == VP8_STATUS_OK, "WebPGetFeatures failed with error code ", res);
  // TODO_IMAGE: support animated webp files, returning an (N, C, H, W) tensor
  // like the plan calls for. This needs the WebPAnimDecoder API (libwebpdemux).
  STD_TORCH_CHECK(
      !features.has_animation, "Animated webp files are not supported.");

  auto return_rgb =
      should_return_rgb(static_cast<ImageReadMode>(mode), features.has_alpha);

  auto decoding_func = return_rgb ? WebPDecodeRGB : WebPDecodeRGBA;
  int num_output_channels = return_rgb ? 3 : 4;

  int width = 0;
  int height = 0;
  auto decoded_data = decoding_func(input_ptr, input_size, &width, &height);
  STD_TORCH_CHECK(decoded_data != nullptr, "WebPDecodeRGB[A] failed.");

  auto deleter = [decoded_data](void*) { WebPFree(decoded_data); };
  auto output = torch::stable::from_blob(
      decoded_data,
      {height, width, num_output_channels},
      {width * num_output_channels, num_output_channels, 1},
      StableDevice(kStableCPU),
      kStableUInt8,
      deleter);

  return stable_permute(output, {2, 0, 1});
}

} // namespace facebook::torchcodec

#endif // !WEBP_FOUND
