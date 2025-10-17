// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/Transform.h"
#include <torch/types.h>
#include "src/torchcodec/_core/FFMPEGCommon.h"

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

int toSwsInterpolation(ResizeTransform::InterpolationMode mode) {
  switch (mode) {
    case ResizeTransform::InterpolationMode::BILINEAR:
      return SWS_BILINEAR;
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
      ":sws_flags=" + toFilterGraphInterpolation(interpolationMode_);
}

std::optional<FrameDims> ResizeTransform::getOutputFrameDims() const {
  return outputDims_;
}

bool ResizeTransform::isResize() const {
  return true;
}

int ResizeTransform::getSwsFlags() const {
  return toSwsInterpolation(interpolationMode_);
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

void CropTransform::validate(const StreamMetadata& streamMetadata) const {
  TORCH_CHECK(x_ <= streamMetadata.width, "Crop x position out of bounds");
  TORCH_CHECK(
      x_ + outputDims_.width <= streamMetadata.width,
      "Crop x position out of bounds")
  TORCH_CHECK(y_ <= streamMetadata.height, "Crop y position out of bounds");
  TORCH_CHECK(
      y_ + outputDims_.height <= streamMetadata.height,
      "Crop y position out of bounds");
}

} // namespace facebook::torchcodec
