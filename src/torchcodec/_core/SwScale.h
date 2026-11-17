// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

struct FrameDims;

struct SwScaleOptions {
  // Flags of the resize.
  int resize_sws_flags = SWS_BILINEAR;
  // Resize in the input pixel format, in the color conversion pass.
  bool resize_before_color_conversion = false;
};

// By default, SwScale uses a double swscale path:
// 1. Color conversion (e.g., YUV -> RGB24/RGB48) at the original frame
//    resolution
// 2. Resize in output RGB space (if resizing is needed)
//
// This approach ensures that transforms happen in the output color space
// (RGB) rather than the input color space (YUV).
//
// With resize_before_color_conversion, a needed resize happens in a single
// pass instead: the frame is resized in its input pixel format and converted
// to RGB at the output size. That is cheaper when shrinking frames, but the
// output differs slightly from resizing in RGB.
//
// The caller is responsible for caching SwScale instances and recreating them
// when the context changes, similar to how FilterGraph is managed.
class SwScale {
 public:
  // config.outputFormat is AV_PIX_FMT_RGB24 for 8-bit, AV_PIX_FMT_RGB48 for
  // >8-bit.
  SwScale(const SwsConfig& config, const SwScaleOptions& options = {});

  int convert(const AVFrame& av_frame, torch::stable::Tensor& output_tensor);

  const SwsConfig& get_config() const {
    return config_;
  }

 private:
  SwsConfig config_;
  // Whether a separate resize pass is needed.
  bool needs_resize_;

  // Color conversion context (input format -> output RGB at original
  // resolution, or at the output resolution in the single-pass case).
  UniqueSwsContext color_conversion_sws_context_;

  // Resize context (output RGB at input res -> output RGB at output res).
  // May be null if no resize is needed.
  UniqueSwsContext resize_sws_context_;

  // Scratch buffer holding the result of the color conversion, which is then
  // the input of the resize. Null if no resize is needed.
  // This must be an AVFrame allocated by FFmpeg. We used to allocate this
  // ourselves as a tensor, but that could cause a crash in some rare cases:
  // swscale may read **past** the source passed to sws_scale() when it's using
  // SIMD ops. It typically reads past the linesize on each row, which is fine
  // for all rows except the last one, where this is at best UB, at worst a
  // segfault. IMHO this qualifies as a bug in swscale, and FWIW we fixed the
  // exact same familiy of bugs in torch
  // https://github.com/pytorch/pytorch/pull/179814!
  // There's no non-regression test for that because it'd be too difficult to
  // trigger, but context can be found in D120434423 and
  // https://fburl.com/phabricator/7yqql5lj.
  UniqueAVFrame color_converted_frame_;
};

} // namespace facebook::torchcodec
