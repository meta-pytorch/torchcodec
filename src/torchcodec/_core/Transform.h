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
  virtual std::string getFilterGraphCpu() const = 0;
  virtual ~Transform() = default;

  // If the transformation does not change the output frame dimensions, then
  // there is no need to override this member function. The default
  // implementation returns an empty optional, indicating that the output frame
  // has the same dimensions as the input frame.
  //
  // If the transformation does change the output frame dimensions, then it
  // must override this member function and return the output frame dimensions.
  virtual std::optional<FrameDims> getOutputFrameDims() const {
    return std::nullopt;
  }

  // The validity of some transforms depends on the characteristics of the
  // AVStream they're being applied to. For example, some transforms will
  // specify coordinates inside a frame, we need to validate that those are
  // within the frame's bounds.
  //
  // Note that the validation function does not return anything. We expect
  // invalid configurations to throw an exception.
  virtual void validate([[maybe_unused]] const FrameDims& inputDims) const {}
};

class ResizeTransform : public Transform {
 public:
  enum class InterpolationMode { BILINEAR };

  explicit ResizeTransform(const FrameDims& dims)
      : outputDims_(dims), interpolationMode_(InterpolationMode::BILINEAR) {}

  ResizeTransform(const FrameDims& dims, InterpolationMode interpolationMode)
      : outputDims_(dims), interpolationMode_(interpolationMode) {}

  std::string getFilterGraphCpu() const override;
  std::optional<FrameDims> getOutputFrameDims() const override;

 private:
  FrameDims outputDims_;
  InterpolationMode interpolationMode_;
};

class CropTransform : public Transform {
 public:
  CropTransform(const FrameDims& dims, int x, int y);

  // Becomes a center crop if x and y are not specified.
  explicit CropTransform(const FrameDims& dims);

  std::string getFilterGraphCpu() const override;
  std::optional<FrameDims> getOutputFrameDims() const override;
  void validate(const FrameDims& inputDims) const override;

 private:
  FrameDims outputDims_;
  std::optional<int> x_;
  std::optional<int> y_;
};

} // namespace facebook::torchcodec
