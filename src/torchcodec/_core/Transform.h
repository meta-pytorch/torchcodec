// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string>
#include "Frame.h"
#include "Metadata.h"

namespace facebook::torchcodec {

class Transform {
 public:
  virtual std::string get_filter_graph_cpu() const = 0;
  virtual ~Transform() = default;

  // If the transformation does not change the output frame dimensions, then
  // there is no need to override this member function. The default
  // implementation returns an empty optional, indicating that the output frame
  // has the same dimensions as the input frame.
  //
  // If the transformation does change the output frame dimensions, then it
  // must override this member function and return the output frame dimensions.
  virtual std::optional<FrameDims> get_output_frame_dims() const {
    return std::nullopt;
  }

  // The ResizeTransform is special because it is the only transform
  // that swscale can handle.
  virtual bool is_resize() const {
    return false;
  }

  // The validity of some transforms depends on the characteristics of the
  // AVStream they're being applied to. For example, some transforms will
  // specify coordinates inside a frame, we need to validate that those are
  // within the frame's bounds.
  //
  // Note that the validation function does not return anything. We expect
  // invalid configurations to throw an exception.
  virtual void validate([[maybe_unused]] const FrameDims& input_dims) const {}
};

class ResizeTransform : public Transform {
 public:
  enum class InterpolationMode { BILINEAR };

  explicit ResizeTransform(const FrameDims& dims)
      : output_dims_(dims), interpolation_mode_(InterpolationMode::BILINEAR) {}

  ResizeTransform(const FrameDims& dims, InterpolationMode interpolation_mode)
      : output_dims_(dims), interpolation_mode_(interpolation_mode) {}

  std::string get_filter_graph_cpu() const override;
  std::optional<FrameDims> get_output_frame_dims() const override;
  bool is_resize() const override;

  int get_sws_flags() const;

 private:
  FrameDims output_dims_;
  InterpolationMode interpolation_mode_;
};

class CropTransform : public Transform {
 public:
  CropTransform(const FrameDims& dims, int x, int y);

  // Becomes a center crop if x and y are not specified.
  explicit CropTransform(const FrameDims& dims);

  std::string get_filter_graph_cpu() const override;
  std::optional<FrameDims> get_output_frame_dims() const override;
  void validate(const FrameDims& input_dims) const override;

 private:
  FrameDims output_dims_;
  std::optional<int> x_;
  std::optional<int> y_;
};

// Rotation values for RotationTransform.
// These correspond to video metadata rotation angles.
enum class Rotation {
  NONE, // 0°
  CCW90, // 90° counter-clockwise
  CW90, // 90° clockwise (or -90°)
  ROTATE180 // 180° (or -180°)
};

// Converts rotation degrees from video metadata to Rotation enum.
// Input is expected in the range [-180, 180].
// Rounds to nearest multiple of 90 degrees before converting.
// Returns Rotation::NONE for nullopt.
Rotation rotation_from_degrees(std::optional<double> degrees);

// Applies rotation in multiples of 90 degrees using FFmpeg's transpose/flip
// filters. Note: this does not support arbitrary angle rotation
// like TorchVision's RandomRotation transform.
// Handles rotation in the filter graph so that user transforms
// operate in post-rotation coordinate space.
class RotationTransform : public Transform {
 public:
  RotationTransform(Rotation rotation, const FrameDims& input_dims);

  std::string get_filter_graph_cpu() const override;
  std::optional<FrameDims> get_output_frame_dims() const override;

 private:
  Rotation rotation_;
  FrameDims output_dims_;
};

} // namespace facebook::torchcodec
