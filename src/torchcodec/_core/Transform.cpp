// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Transform.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

namespace {

std::string to_filter_graph_interpolation(
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

int to_sws_interpolation(ResizeTransform::InterpolationMode mode) {
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

std::string ResizeTransform::get_filter_graph_cpu() const {
  return "scale=" + std::to_string(output_dims_.width) + ":" +
      std::to_string(output_dims_.height) +
      ":flags=" + to_filter_graph_interpolation(interpolation_mode_);
}

std::optional<FrameDims> ResizeTransform::get_output_frame_dims() const {
  return output_dims_;
}

bool ResizeTransform::is_resize() const {
  return true;
}

int ResizeTransform::get_sws_flags() const {
  return to_sws_interpolation(interpolation_mode_);
}

CropTransform::CropTransform(const FrameDims& dims) : output_dims_(dims) {}

CropTransform::CropTransform(const FrameDims& dims, int x, int y)
    : output_dims_(dims), x_(x), y_(y) {
  STD_TORCH_CHECK(x >= 0, "Crop x position must be >= 0, got: ", x);
  STD_TORCH_CHECK(y >= 0, "Crop y position must be >= 0, got: ", y);
}

std::string CropTransform::get_filter_graph_cpu() const {
  // For the FFmpeg filter crop, if the x and y coordinates are left
  // unspecified, it defaults to a center crop.
  std::string coordinates = x_.has_value()
      ? (":" + std::to_string(x_.value()) + ":" + std::to_string(y_.value()))
      : "";
  return "crop=" + std::to_string(output_dims_.width) + ":" +
      std::to_string(output_dims_.height) + coordinates + ":exact=1";
}

std::optional<FrameDims> CropTransform::get_output_frame_dims() const {
  return output_dims_;
}

void CropTransform::validate(const FrameDims& input_dims) const {
  STD_TORCH_CHECK(
      output_dims_.height <= input_dims.height,
      "Crop output height (",
      output_dims_.height,
      ") is greater than input height (",
      input_dims.height,
      ")");
  STD_TORCH_CHECK(
      output_dims_.width <= input_dims.width,
      "Crop output width (",
      output_dims_.width,
      ") is greater than input width (",
      input_dims.width,
      ")");
  STD_TORCH_CHECK(
      x_.has_value() == y_.has_value(),
      "Crop x and y values must be both set or both unset");
  if (x_.has_value()) {
    STD_TORCH_CHECK(
        x_.value() <= input_dims.width,
        "Crop x start position, ",
        x_.value(),
        ", out of bounds of input width, ",
        input_dims.width);
    STD_TORCH_CHECK(
        x_.value() + output_dims_.width <= input_dims.width,
        "Crop x end position, ",
        x_.value() + output_dims_.width,
        ", out of bounds of input width ",
        input_dims.width);
    STD_TORCH_CHECK(
        y_.value() <= input_dims.height,
        "Crop y start position, ",
        y_.value(),
        ", out of bounds of input height, ",
        input_dims.height);
    STD_TORCH_CHECK(
        y_.value() + output_dims_.height <= input_dims.height,
        "Crop y end position, ",
        y_.value() + output_dims_.height,
        ", out of bounds of input height ",
        input_dims.height);
  }
}

Rotation rotation_from_degrees(std::optional<double> degrees) {
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
      STD_TORCH_CHECK(
          false,
          "Unexpected rotation value: ",
          *degrees,
          ". Expected range is [-180, 180].");
  }
}

RotationTransform::RotationTransform(
    Rotation rotation,
    const FrameDims& input_dims)
    : rotation_(rotation) {
  // 90° rotations swap dimensions
  if (rotation_ == Rotation::CCW90 || rotation_ == Rotation::CW90) {
    output_dims_ = FrameDims(input_dims.width, input_dims.height);
  } else {
    output_dims_ = input_dims;
  }
}

std::string RotationTransform::get_filter_graph_cpu() const {
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

std::optional<FrameDims> RotationTransform::get_output_frame_dims() const {
  return output_dims_;
}

} // namespace facebook::torchcodec
