// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Transform.h"
#include "StableABICompat.h"
#include "FFMPEGCommon.h"

namespace facebook::torchcodec {

namespace {

std::string toFilterGraphInterpolation(
    ResizeTransform::InterpolationMode mode) {
  switch (mode) {
    case ResizeTransform::InterpolationMode::BILINEAR:
      return "bilinear";
    default:
      STABLE_CHECK(
          false,
          "Unknown interpolation mode: " +
              std::to_string(static_cast<int>(mode)));
      return ""; // unreachable, but silences compiler warning
  }
}

int toSwsInterpolation(ResizeTransform::InterpolationMode mode) {
  switch (mode) {
    case ResizeTransform::InterpolationMode::BILINEAR:
      return SWS_BILINEAR;
    default:
      STABLE_CHECK(
          false,
          "Unknown interpolation mode: " +
              std::to_string(static_cast<int>(mode)));
      return 0; // unreachable, but silences compiler warning
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
  STABLE_CHECK(x >= 0, "Crop x position must be >= 0, got: " + std::to_string(x));
  STABLE_CHECK(y >= 0, "Crop y position must be >= 0, got: " + std::to_string(y));
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
  STABLE_CHECK(
      outputDims_.height <= inputDims.height,
      "Crop output height (" + std::to_string(outputDims_.height) +
          ") is greater than input height (" + std::to_string(inputDims.height) + ")");
  STABLE_CHECK(
      outputDims_.width <= inputDims.width,
      "Crop output width (" + std::to_string(outputDims_.width) +
          ") is greater than input width (" + std::to_string(inputDims.width) + ")");
  STABLE_CHECK(
      x_.has_value() == y_.has_value(),
      "Crop x and y values must be both set or both unset");
  if (x_.has_value()) {
    STABLE_CHECK(
        x_.value() <= inputDims.width,
        "Crop x start position, " + std::to_string(x_.value()) +
            ", out of bounds of input width, " + std::to_string(inputDims.width));
    STABLE_CHECK(
        x_.value() + outputDims_.width <= inputDims.width,
        "Crop x end position, " + std::to_string(x_.value() + outputDims_.width) +
            ", out of bounds of input width " + std::to_string(inputDims.width));
    STABLE_CHECK(
        y_.value() <= inputDims.height,
        "Crop y start position, " + std::to_string(y_.value()) +
            ", out of bounds of input height, " + std::to_string(inputDims.height));
    STABLE_CHECK(
        y_.value() + outputDims_.height <= inputDims.height,
        "Crop y end position, " + std::to_string(y_.value() + outputDims_.height) +
            ", out of bounds of input height " + std::to_string(inputDims.height));
  }
}

} // namespace facebook::torchcodec
