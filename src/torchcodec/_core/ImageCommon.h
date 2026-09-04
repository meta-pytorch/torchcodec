// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

namespace facebook::torchcodec {

// Must be kept in-sync with the Python ImageReadMode enum in
// torchcodec/decoders/_image_decoders.py (and matching torchvision's
// ImageReadMode).
enum class ImageReadMode : int64_t {
  UNCHANGED = 0,
  GRAY = 1,
  GRAY_ALPHA = 2,
  RGB = 3,
  RGB_ALPHA = 4,
};

// Requested output dtype. Must be kept in-sync with the codes in
// _validate_output_dtype in torchcodec/decoders/_image_decoders.py. Only
// relevant for the decoders whose source can carry more than 8 bits per channel
// (PNG, AVIF, HEIC); AUTO keeps the source's native precision (16-bit for
// >8-bit sources, 8-bit otherwise).
//
// The image counterpart of OutputDtypeConfig (StreamOptions.h), which the video
// side resolves in the same way against its own source. The two aren't shared
// because the dtypes they choose between differ: uint8/uint16 here against
// uint8/float32 there.
enum class ImageOutputDtypeConfig : int64_t {
  UINT8 = 0,
  UINT16 = 1,
  AUTO = 2,
};

// Whether a decoder should output 8bit or 16bit data based on the user's
// requeted dtype, and the source. This is assumed to be called on a decoder
// that supports >8bit sources.
inline bool should_output_uint16(
    ImageOutputDtypeConfig output_dtype_config,
    bool source_gt_8bit) {
  switch (output_dtype_config) {
    case ImageOutputDtypeConfig::UINT8:
      return false;
    case ImageOutputDtypeConfig::UINT16:
      return true;
    case ImageOutputDtypeConfig::AUTO:
      return source_gt_8bit;
    default:
      STD_TORCH_CHECK(
          false,
          "Unexpected output_dtype ",
          static_cast<int64_t>(output_dtype_config),
          ". This should never happen, please report a bug to the TorchCodec repo.");
  }
}

// Whether a decoder should produce a 3-channel RGB tensor (true) or a 4-channel
// RGBA one (false) for the given read mode. `has_alpha` is whether the source
// actually carries transparency. Only RGB, RGB_ALPHA and UNCHANGED are handled:
// the grayscale modes are emulated in Python (see _decode_to_mode() in
// _image_decoders.py), which requests RGB/RGBA from the C++ decoders and
// converts, so the default branch below is unreachable in practice. Shared by
// the decoders whose native output is RGB/RGBA (e.g. webp, gif).
inline bool should_return_rgb(ImageReadMode mode, bool has_alpha) {
  switch (mode) {
    case ImageReadMode::RGB:
      return true;
    case ImageReadMode::RGB_ALPHA:
      return false;
    case ImageReadMode::UNCHANGED:
      return !has_alpha;
    default:
      STD_TORCH_CHECK(
          false,
          "Reached an unexpected code path while decoding an image to mode ",
          static_cast<int64_t>(mode),
          ". This should never happen, please report a bug to the TorchCodec repo.");
  }
}

// Validates that `input` is a non-empty, 1-dimensional, uint8 CPU tensor of
// encoded image bytes, and returns a contiguous version of it.
inline torch::stable::Tensor validate_encoded_data(
    const torch::stable::Tensor& input) {
  STD_TORCH_CHECK(
      input.device().type() == kStableCPU,
      "Input tensor must be on the CPU, got a tensor on ",
      device_type_name(input.device().type()),
      ".");
  STD_TORCH_CHECK(
      input.scalar_type() == kStableUInt8,
      "Input tensor must have uint8 data type, got ",
      torch::headeronly::toString(input.scalar_type()));
  STD_TORCH_CHECK(
      input.dim() == 1 && input.numel() > 0,
      "Input tensor must be 1-dimensional and non-empty, got ",
      input.dim(),
      " dims  and ",
      input.numel(),
      " numels.");
  return torch::stable::contiguous(input);
}

} // namespace facebook::torchcodec
