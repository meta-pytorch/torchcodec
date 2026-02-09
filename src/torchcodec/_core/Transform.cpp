// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Transform.h"
#include <torch/types.h>
#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

namespace {

std::string toFilterGraphInterpolation(
    ResizeTransform::InterpolationMode mode) {
  switch (mode) {
    case ResizeTransform::InterpolationMode::BILINEAR:
      return "bilinear";
    default:
      STD_TORCH_CHECK(
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
      STD_TORCH_CHECK(
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

bool ResizeTransform::isResize() const {
  return true;
}

int ResizeTransform::getSwsFlags() const {
  return toSwsInterpolation(interpolationMode_);
}

CropTransform::CropTransform(const FrameDims& dims) : outputDims_(dims) {}

CropTransform::CropTransform(const FrameDims& dims, int x, int y)
    : outputDims_(dims), x_(x), y_(y) {
  STD_TORCH_CHECK(x >= 0, "Crop x position must be >= 0, got: ", x);
  STD_TORCH_CHECK(y >= 0, "Crop y position must be >= 0, got: ", y);
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
  STD_TORCH_CHECK(
      outputDims_.height <= inputDims.height,
      "Crop output height (",
      outputDims_.height,
      ") is greater than input height (",
      inputDims.height,
      ")");
  STD_TORCH_CHECK(
      outputDims_.width <= inputDims.width,
      "Crop output width (",
      outputDims_.width,
      ") is greater than input width (",
      inputDims.width,
      ")");
  STD_TORCH_CHECK(
      x_.has_value() == y_.has_value(),
      "Crop x and y values must be both set or both unset");
  if (x_.has_value()) {
    STD_TORCH_CHECK(
        x_.value() <= inputDims.width,
        "Crop x start position, ",
        x_.value(),
        ", out of bounds of input width, ",
        inputDims.width);
    STD_TORCH_CHECK(
        x_.value() + outputDims_.width <= inputDims.width,
        "Crop x end position, ",
        x_.value() + outputDims_.width,
        ", out of bounds of input width ",
        inputDims.width);
    STD_TORCH_CHECK(
        y_.value() <= inputDims.height,
        "Crop y start position, ",
        y_.value(),
        ", out of bounds of input height, ",
        inputDims.height);
    STD_TORCH_CHECK(
        y_.value() + outputDims_.height <= inputDims.height,
        "Crop y end position, ",
        y_.value() + outputDims_.height,
        ", out of bounds of input height ",
        inputDims.height);
  }
}

Rotation rotationFromDegrees(std::optional<double> degrees) {
  if (!degrees.has_value()) {
    return Rotation::NONE;
  }
  // Round to nearest multiple of 90 degrees
  int rounded = static_cast<int>(std::round(*degrees / 90.0)) * 90;
  switch (rounded) {
    case 0:
      return Rotation::NONE;
    case 90:
      return Rotation::CCW90;
    case -90:
      return Rotation::CW90;
    case 180:
    case -180:
      return Rotation::ROTATE180;
    default:
      TORCH_CHECK(
          false,
          "Unexpected rotation value: ",
          *degrees,
          ". Expected range is [-180, 180].");
  }
}

RotationTransform::RotationTransform(
    Rotation rotation,
    const FrameDims& inputDims)
    : rotation_(rotation) {
  // 90° rotations swap dimensions
  if (rotation_ == Rotation::CCW90 || rotation_ == Rotation::CW90) {
    outputDims_ = FrameDims(inputDims.width, inputDims.height);
  } else {
    outputDims_ = inputDims;
  }
}

std::string RotationTransform::getFilterGraphCpu() const {
  switch (rotation_) {
    case Rotation::NONE:
      return "";
    case Rotation::CCW90:
      return "transpose=cclock";
    case Rotation::CW90:
      return "transpose=clock";
    case Rotation::ROTATE180:
      return "hflip,vflip";
    default:
      return "";
  }
}

std::optional<FrameDims> RotationTransform::getOutputFrameDims() const {
  return outputDims_;
}

} // namespace facebook::torchcodec
