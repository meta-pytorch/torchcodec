// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include "StableABICompat.h"

namespace facebook::torchcodec {

enum ColorConversionLibrary {
  // Use the libavfilter library for color conversion.
  FILTERGRAPH,
  // Use the libswscale library for color conversion.
  SWSCALE
};

// Controls the dtype of decoded frame tensors.
// UINT8: Always output uint8 tensors (default, backward compatible).
// FLOAT32: Always output float32 tensors normalized to [0, 1].
// AUTO: Output uint8 for SDR (<=8-bit) sources, float32 for HDR (>8-bit).
enum class OutputDtype { UINT8, FLOAT32, AUTO };

// Returns the effective output bit depth given the source bit depth and the
// user's OutputDtype setting.
// UINT8: always 8. FLOAT32: preserve source. AUTO: 8 for <=8-bit, else source.
inline int resolvedBitDepth(int sourceBitDepth, OutputDtype outputDtype) {
  switch (outputDtype) {
    case OutputDtype::UINT8:
      return 8;
    case OutputDtype::FLOAT32:
      return sourceBitDepth;
    case OutputDtype::AUTO:
      return (sourceBitDepth <= 8) ? 8 : sourceBitDepth;
  }
  return 8;
}

struct VideoStreamOptions {
  VideoStreamOptions() {}

  // Number of threads we pass to FFMPEG for decoding.
  // 0 means FFMPEG will choose the number of threads automatically to fully
  // utilize all cores. If not set, it will be the default FFMPEG behavior for
  // the given codec.
  std::optional<int> ffmpegThreadCount;

  // Currently the dimension order can be either NHWC or NCHW.
  // H=height, W=width, C=channel.
  std::string dimensionOrder = "NCHW";

  // By default we have to use filtergraph, as it is more general. We can only
  // use swscale when we have met strict requirements. See
  // CpuDeviceInterface::initialze() for the logic.
  ColorConversionLibrary colorConversionLibrary =
      ColorConversionLibrary::FILTERGRAPH;

  // By default we use CPU for decoding for both C++ and python users.
  // Note: This is not used for video encoding, because device is determined by
  // the device of the input frame tensor.
  StableDevice device = StableDevice(kStableCPU);
  // Device variant (e.g., "ffmpeg", "beta", etc.)
  std::string_view deviceVariant = "ffmpeg";

  // Controls the dtype of decoded frame tensors. Default UINT8 preserves
  // existing behavior; FLOAT32 and AUTO are used for high-bit-depth output
  // (e.g. HDR).
  OutputDtype outputDtype = OutputDtype::UINT8;

  // Encoding options
  std::optional<std::string> codec;
  // Optional pixel format for video encoding (e.g., "yuv420p", "yuv444p")
  // If not specified, uses codec's default format.
  std::optional<std::string> pixelFormat;
  std::optional<double> crf;
  std::optional<std::string> preset;
  std::optional<std::map<std::string, std::string>> extraOptions;
};

struct AudioStreamOptions {
  AudioStreamOptions() {}

  // Encoding only
  std::optional<int> bitRate;
  // Decoding and encoding:
  std::optional<int> numChannels;
  std::optional<int> sampleRate;
};

} // namespace facebook::torchcodec
