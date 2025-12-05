// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Transform.h"
#include <torch/types.h>
#include "FFMPEGCommon.h"

namespace facebook::torchcodec {

namespace {

std::string toFilterGraphInterpolation(
    ResizeTransform::InterpolationMode mode) {
  switch (mode) {
    case ResizeTransform::InterpolationMode::BILINEAR:
      return "bilinear";
    default:
      TORCH_CHECK(
          false,
          "Unknown interpolation mode: " +
              std::to_string(static_cast<int>(mode)));
  }
}

} // namespace

std::string ResizeTransform::getFilterGraphCpu() const {
  return "scale=" + std::to_string(outputDims_.width) + ":" +
      std::to_string(outputDims_.height) +
      ":flags=" + toFilterGraphInterpolation(interpolationMode_);
}

std::optional<FrameDims> ResizeTransform::getOutputFrameDims() const {
  return outputDims_;
}

CropTransform::CropTransform(const FrameDims& dims) : outputDims_(dims) {}

CropTransform::CropTransform(const FrameDims& dims, int x, int y)
    : outputDims_(dims), x_(x), y_(y) {
  TORCH_CHECK(x_ >= 0, "Crop x position must be >= 0, got: ", x_);
  TORCH_CHECK(y_ >= 0, "Crop y position must be >= 0, got: ", y_);
}

std::string CropTransform::getFilterGraphCpu() const {
  // For the FFmpeg filter crop, if the x and y coordinates are left
  // unspecified, it defaults to a center crop.
  std::string coordinates = x_.has_value()
      ? (":" + std::to_string(x_.value()) + ":" + std::to_string(y_.value()))
      : "";
  return "crop=" + std::to_string(outputDims_.width) + ":" +
      std::to_string(outputDims_.height) + coordinates + ":exact=1";
}

std::optional<FrameDims> CropTransform::getOutputFrameDims() const {
  return outputDims_;
}

void CropTransform::validate(const FrameDims& inputDims) const {
  TORCH_CHECK(
      outputDims_.height <= inputDims.height,
      "Crop output height (",
      outputDims_.height,
      ") is greater than input height (",
      inputDims.height,
      ")");
  TORCH_CHECK(
      outputDims_.width <= inputDims.width,
      "Crop output width (",
      outputDims_.width,
      ") is greater than input width (",
      inputDims.width,
      ")");
  TORCH_CHECK(
      x_.has_value() == y_.has_value(),
      "Crop x and y values must be both set or both unset");
  if (x_.has_value()) {
    TORCH_CHECK(
        x_.value() <= inputDims.width,
        "Crop x start position, ",
        x_.value(),
        ", out of bounds of input width, ",
        inputDims.width);
    TORCH_CHECK(
        x_.value() + outputDims_.width <= inputDims.width,
        "Crop x end position, ",
        x_.value() + outputDims_.width,
        ", out of bounds of input width ",
        inputDims.width);
    TORCH_CHECK(
        y_.value() <= inputDims.height,
        "Crop y start position, ",
        y_.value(),
        ", out of bounds of input height, ",
        inputDims.height);
    TORCH_CHECK(
        y_.value() + outputDims_.height <= inputDims.height,
        "Crop y end position, ",
        y_.value() + outputDims_.height,
        ", out of bounds of input height ",
        inputDims.height);
  }
}

} // namespace facebook::torchcodec
