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

CropTransform::CropTransform(const FrameDims& dims, int x, int y)
    : outputDims_(dims), x_(x), y_(y) {
  TORCH_CHECK(x_ >= 0, "Crop x position must be >= 0, got: ", x_);
  TORCH_CHECK(y_ >= 0, "Crop y position must be >= 0, got: ", y_);
}

std::string CropTransform::getFilterGraphCpu() const {
  return "crop=" + std::to_string(outputDims_.width) + ":" +
      std::to_string(outputDims_.height) + ":" + std::to_string(x_) + ":" +
      std::to_string(y_) + ":exact=1";
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
      x_ <= inputDims.width,
      "Crop x start position, ",
      x_,
      ", out of bounds of input width, ",
      inputDims.width);
  TORCH_CHECK(
      x_ + outputDims_.width <= inputDims.width,
      "Crop x end position, ",
      x_ + outputDims_.width,
      ", out of bounds of input width ",
      inputDims.width);
  TORCH_CHECK(
      y_ <= inputDims.height,
      "Crop y start position, ",
      y_,
      ", out of bounds of input height, ",
      inputDims.height);
  TORCH_CHECK(
      y_ + outputDims_.height <= inputDims.height,
      "Crop y end position, ",
      y_ + outputDims_.height,
      ", out of bounds of input height ",
      inputDims.height);
}

} // namespace facebook::torchcodec
