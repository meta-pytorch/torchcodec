// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"
#include "StreamOptions.h"

namespace facebook::torchcodec {

class FORCE_PUBLIC_VISIBILITY ColorConverter {
 public:
  explicit ColorConverter(
      const StableDevice& device = StableDevice(kStableCPU),
      OutputDtypeConfig output_dtype_config = OutputDtypeConfig::UINT8);

  // `frame_device` is where the frame's samples live. It must be this
  // converter's device: we refuse to move samples around behind your back.
  torch::stable::Tensor convert(
      const AVFrame& av_frame,
      const StableDevice& frame_device);

 private:
  void maybe_initialize_interface(OutputDtype output_dtype);

  std::unique_ptr<DeviceInterface> device_interface_;
  StableDevice device_;
  OutputDtypeConfig output_dtype_config_;
  std::optional<OutputDtype> initialized_output_dtype_;
};

} // namespace facebook::torchcodec
