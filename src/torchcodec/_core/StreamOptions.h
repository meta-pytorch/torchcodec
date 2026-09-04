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

// What a frame is actually converted to. Derived, never configured: it is
// resolve_output_dtype(config, source pixel format), so it can only be known
// once the source is. Passed to whoever converts or allocates; deliberately not
// a field of VideoStreamOptions, which describes what was asked for rather than
// what was worked out from it.
enum class OutputDtype { UINT8, FLOAT32 };

// The user-facing output dtype config, resolved into an OutputDtype by
// resolve_output_dtype().
// UINT8: Always output uint8 tensors (default, backward compatible). Uses an
//        8-bit / RGB24 intermediate.
// FLOAT32: Always output float32 tensors normalized to [0, 1]. Uses a 16-bit /
//          RGB48 intermediate so the YUV->RGB matrix output is preserved at
//          full precision through the float cast, regardless of source bit
//          depth.
// AUTO: Output uint8 for SDR (<=8-bit) sources, float32 for HDR (>8-bit).
enum class OutputDtypeConfig { UINT8, FLOAT32, AUTO };

struct VideoStreamOptions {
  VideoStreamOptions() {}

  // Number of threads we pass to FFMPEG for decoding.
  // 0 means FFMPEG will choose the number of threads automatically to fully
  // utilize all cores. If not set, it will be the default FFMPEG behavior for
  // the given codec.
  std::optional<int> ffmpeg_thread_count;

  // Currently the dimension order can be either NHWC or NCHW.
  // H=height, W=width, C=channel.
  std::string dimension_order = "NCHW";

  // By default we have to use filtergraph, as it is more general. We can only
  // use swscale when we have met strict requirements. See
  // CpuDeviceInterface::initialze() for the logic.
  ColorConversionLibrary color_conversion_library =
      ColorConversionLibrary::FILTERGRAPH;

  // By default we use CPU for decoding for both C++ and python users.
  // Note: This is not used for video encoding, because device is determined by
  // the device of the input frame tensor.
  StableDevice device = StableDevice(kStableCPU);
  // Device variant (e.g., "nvdec", "ffmpeg")
  std::string_view device_variant = "default";

  // What the user asked for. Resolving it needs the source's pixel format, so
  // that happens where the source is known rather than here.
  OutputDtypeConfig output_dtype_config = OutputDtypeConfig::UINT8;

  // Encoding options
  std::optional<std::string> codec;
  // Optional pixel format for video encoding (e.g., "yuv420p", "yuv444p")
  // If not specified, uses codec's default format.
  std::optional<std::string> pixel_format;
  std::optional<double> crf;
  std::optional<std::string> preset;
  std::optional<std::map<std::string, std::string>> extra_options;
};

struct AudioStreamOptions {
  AudioStreamOptions() {}

  // Encoding only
  std::optional<int> bit_rate;
  // Decoding and encoding:
  std::optional<int> num_channels;
  std::optional<int> sample_rate;
};

} // namespace facebook::torchcodec
