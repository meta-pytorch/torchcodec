// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecodeAvif.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !AVIF_FOUND

namespace facebook::torchcodec {

torch::stable::Tensor decode_avif(
    [[maybe_unused]] const torch::stable::Tensor& data,
    [[maybe_unused]] int64_t mode) {
  STD_TORCH_CHECK(
      false,
      "decode_avif: torchcodec was not compiled with libavif support. "
      "Rebuild torchcodec in an environment where libavif (and its development "
      "headers) are available. If you see this error in a prebuilt wheel, "
      "please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

#include <cstring>
#include <memory>

#include "avif/avif.h"

#include "ImageCommon.h"

namespace facebook::torchcodec {

namespace {

// This normally comes from avif_cxx.h, but that header isn't always present
// when installing libavif, so we define the deleter ourselves.
struct AvifDecoderDeleter {
  void operator()(avifDecoder* decoder) const {
    avifDecoderDestroy(decoder);
  }
};

using DecoderPtr = std::unique_ptr<avifDecoder, AvifDecoderDeleter>;

} // namespace

torch::stable::Tensor decode_avif(
    const torch::stable::Tensor& input,
    int64_t mode) {
  // Based on
  // https://github.com/AOMediaCodec/libavif/blob/main/examples/avif_example_decode_memory.c
  validate_encoded_data(input);

  DecoderPtr decoder(avifDecoderCreate());
  STD_TORCH_CHECK(decoder != nullptr, "Failed to create avif decoder.");

  auto result = avifDecoderSetIOMemory(
      decoder.get(), input.const_data_ptr<uint8_t>(), input.numel());
  STD_TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifDecoderSetIOMemory failed: ",
      avifResultToString(result));

  result = avifDecoderParse(decoder.get());
  STD_TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifDecoderParse failed: ",
      avifResultToString(result));
  // TODO_IMAGE: support animated AVIF files, returning an (N, C, H, W) tensor
  // like the plan calls for (see the GIF decoder for the multi-frame pattern).
  STD_TORCH_CHECK(
      decoder->imageCount == 1, "Animated AVIF files are not supported.");

  result = avifDecoderNextImage(decoder.get());
  STD_TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifDecoderNextImage failed: ",
      avifResultToString(result));

  avifRGBImage rgb;
  std::memset(&rgb, 0, sizeof(rgb));
  avifRGBImageSetDefaults(&rgb, decoder->image);

  // TODO_IMAGE: images encoded as 10 or 12 bits should be decoded as uint16 to
  // preserve their precision, like torchvision does. We force 8 bits for now so
  // AVIF stays consistent with the other torchcodec image decoders, which are
  // all uint8. This is tied to adding 16-bit support to the libpng decoder.
  rgb.depth = 8;

  auto return_rgb = should_return_rgb(
      static_cast<ImageReadMode>(mode),
      static_cast<bool>(decoder->alphaPresent));

  int num_channels = return_rgb ? 3 : 4;
  rgb.format = return_rgb ? AVIF_RGB_FORMAT_RGB : AVIF_RGB_FORMAT_RGBA;
  rgb.ignoreAlpha = return_rgb ? AVIF_TRUE : AVIF_FALSE;

  auto output = torch::stable::empty(
      {static_cast<int64_t>(rgb.height),
       static_cast<int64_t>(rgb.width),
       num_channels},
      kStableUInt8);
  rgb.pixels = static_cast<uint8_t*>(output.mutable_data_ptr());
  rgb.rowBytes = rgb.width * avifRGBImagePixelSize(&rgb);

  result = avifImageYUVToRGB(decoder->image, &rgb);
  STD_TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifImageYUVToRGB failed: ",
      avifResultToString(result));

  // TODO_IMAGE: apply AVIF orientation (irot/imir transforms) like we apply
  // EXIF orientation for the other decoders.
  return stable_permute(output, {2, 0, 1}); // HWC -> CHW
}

} // namespace facebook::torchcodec

#endif // !AVIF_FOUND
