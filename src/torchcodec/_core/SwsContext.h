// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

extern "C" {
#include <libswscale/swscale.h>
}

#include "src/torchcodec/_core/Frame.h"

namespace facebook::torchcodec {

// Context describing frame properties needed for swscale conversion.
// Used to detect when swscale context needs to be recreated.
struct SwsFrameContext {
  int inputWidth;
  int inputHeight;
  AVPixelFormat inputFormat;
  int outputWidth;
  int outputHeight;

  SwsFrameContext(
      int inputWidth,
      int inputHeight,
      AVPixelFormat inputFormat,
      int outputWidth,
      int outputHeight);

  bool operator==(const SwsFrameContext& other) const;
  bool operator!=(const SwsFrameContext& other) const;
};

// Manages swscale context creation and caching across multiple frame conversions.
// Reuses the context as long as frame properties remain the same.
class SwsScaler {
 public:
  SwsScaler() = default;
  ~SwsScaler() = default;

  // Get or create a swscale context for the given frame and output dimensions.
  // Reuses cached context if frame properties haven't changed.
  // Returns a raw pointer to the internal swscale context. The pointer is valid
  // as long as this SwsScaler object is alive.
  SwsContext* getOrCreateContext(
      const UniqueAVFrame& avFrame,
      const FrameDims& outputDims,
      AVColorSpace colorspace,
      AVPixelFormat outputFormat,
      int swsFlags = SWS_BILINEAR);

 private:
  UniqueSwsContext swsContext_;
  SwsFrameContext prevFrameContext_ = SwsFrameContext(0, 0, AV_PIX_FMT_NONE, 0, 0);
};

} // namespace facebook::torchcodec
