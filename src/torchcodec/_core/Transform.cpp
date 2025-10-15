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

std::string CropTransform::getFilterGraphCpu() const {
  // Force format conversion to RGB before cropping to ensure consistent
  // behavior with torchvision and to avoid issues with chroma subsampling in
  // YUV formats.
  return "crop=" + std::to_string(outputDims_.width) + ":" +
      std::to_string(outputDims_.height) + ":" + std::to_string(x_) + ":" +
      std::to_string(y_) + ":exact=1";
}

std::optional<FrameDims> CropTransform::getOutputFrameDims() const {
  return outputDims_;
}

} // namespace facebook::torchcodec
