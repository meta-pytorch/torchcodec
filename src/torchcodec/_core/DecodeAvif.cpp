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

#if !TORCHCODEC_ENABLE_AVIF

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

#include <torch/headeronly/core/MemoryFormat.h>

#include <cstring>
#include <memory>

#include "avif/avif.h"

#include "Exif.h"
#include "ImageCommon.h"

namespace facebook::torchcodec {

using namespace exif_private;

namespace {

// This normally comes from avif_cxx.h, but that header isn't always present
// when installing libavif, so we define the deleter ourselves.
struct AvifDecoderDeleter {
  void operator()(avifDecoder* decoder) const {
    avifDecoderDestroy(decoder);
  }
};

using DecoderPtr = std::unique_ptr<avifDecoder, AvifDecoderDeleter>;

// AVIF stores orientation as separate 'irot' and 'imir' transforms rather than
// a single EXIF orientation value. We map them to the equivalent EXIF
// orientation so we can reuse exif_orientation_transform() and stay consistent
// with the other decoders. This is a port of libavif's own
// avifImageIrotImirToExifOrientation().
ExifOrientation avif_exif_orientation(const avifImage* image) {
  bool has_irot = (image->transformFlags & AVIF_TRANSFORM_IROT) != 0;
  bool has_imir = (image->transformFlags & AVIF_TRANSFORM_IMIR) != 0;
  uint8_t angle = has_irot ? image->irot.angle : 0;
  uint8_t axis = image->imir.axis;

  using E = ExifOrientation;
  switch (angle) {
    case 0:
      return !has_imir ? E::TopLeft : (axis == 0 ? E::BottomLeft : E::TopRight);
    case 1:
      return !has_imir ? E::LeftBottom
                       : (axis == 0 ? E::LeftTop : E::RightBottom);
    case 2:
      return !has_imir ? E::BottomRight
                       : (axis == 0 ? E::TopRight : E::BottomLeft);
    default: // angle == 3
      return !has_imir ? E::RightTop
                       : (axis == 0 ? E::RightBottom : E::LeftTop);
  }
}

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

  // irot/imir are populated during parse and are constant across the sequence.
  ExifOrientation exif_orientation = avif_exif_orientation(decoder->image);

  // We detect animation via imageSequenceTrackPresent (the signal that
  // specifically means "image sequence"), not imageCount. imageCount can also
  // exceed 1 for a *progressive* still images.
  int64_t num_frames =
      decoder->imageSequenceTrackPresent ? decoder->imageCount : 1;

  // alphaPresent is valid after parse and constant across the sequence.
  auto return_rgb = should_return_rgb(
      static_cast<ImageReadMode>(mode),
      static_cast<bool>(decoder->alphaPresent));
  int num_channels = return_rgb ? 3 : 4;

  torch::stable::Tensor output;
  uint8_t* output_ptr = nullptr;
  int64_t frame_num_bytes = 0;

  for (int64_t i = 0; i < num_frames; ++i) {
    result = avifDecoderNextImage(decoder.get());
    STD_TORCH_CHECK(
        result == AVIF_RESULT_OK,
        "avifDecoderNextImage failed at frame ",
        i,
        ": ",
        avifResultToString(result));

    avifRGBImage rgb;
    std::memset(&rgb, 0, sizeof(rgb));
    avifRGBImageSetDefaults(&rgb, decoder->image);

    // TODO_IMAGE: support 10 and 12 bits.
    rgb.depth = 8;
    rgb.format = return_rgb ? AVIF_RGB_FORMAT_RGB : AVIF_RGB_FORMAT_RGBA;
    rgb.ignoreAlpha = return_rgb ? AVIF_TRUE : AVIF_FALSE;

    if (i == 0) {
      output = torch::stable::empty(
          {num_frames,
           num_channels,
           static_cast<int64_t>(rgb.height),
           static_cast<int64_t>(rgb.width)},
          kStableUInt8,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          torch::headeronly::MemoryFormat::ChannelsLast);
      output_ptr = output.mutable_data_ptr<uint8_t>();
      frame_num_bytes =
          static_cast<int64_t>(num_channels) * rgb.height * rgb.width;
    }

    rgb.pixels = output_ptr + i * frame_num_bytes;
    rgb.rowBytes = rgb.width * avifRGBImagePixelSize(&rgb);

    result = avifImageYUVToRGB(decoder->image, &rgb);
    STD_TORCH_CHECK(
        result == AVIF_RESULT_OK,
        "avifImageYUVToRGB failed at frame ",
        i,
        ": ",
        avifResultToString(result));
  }

  output = exif_orientation_transform(output, exif_orientation);

  if (num_frames == 1) {
    output = select_row(output, 0);
  }
  return output;
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_AVIF
