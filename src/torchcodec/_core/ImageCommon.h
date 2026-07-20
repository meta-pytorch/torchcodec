// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

namespace facebook::torchcodec {

// Must be kept in-sync with the Python ImageColorMode enum in
// torchcodec/decoders/_image_decoders.py (and matching torchvision's
// ImageReadMode).
enum class ImageReadMode : int64_t {
  Unchanged = 0,
  Gray = 1,
  GrayAlpha = 2,
  Rgb = 3,
  RgbAlpha = 4,
};

inline void validate_encoded_data(const torch::stable::Tensor& encoded_data) {
  STD_TORCH_CHECK(
      encoded_data.is_contiguous(), "Input tensor must be contiguous.");
  STD_TORCH_CHECK(
      encoded_data.scalar_type() == kStableUInt8,
      "Input tensor must have uint8 data type, got ",
      torch::headeronly::toString(encoded_data.scalar_type()));
  STD_TORCH_CHECK(
      encoded_data.dim() == 1 && encoded_data.numel() > 0,
      "Input tensor must be 1-dimensional and non-empty, got ",
      encoded_data.dim(),
      " dims  and ",
      encoded_data.numel(),
      " numels.");
}

} // namespace facebook::torchcodec
