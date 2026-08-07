// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <optional>
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

  torch::stable::Tensor convert(const AVFrame& av_frame);

 private:
  OutputDtype resolve_output_dtype(const AVFrame& av_frame) const;
  void initialize_for_output_dtype(OutputDtype output_dtype);

  std::unique_ptr<DeviceInterface> device_interface_;
  StableDevice device_;
  OutputDtypeConfig output_dtype_config_;
  std::optional<OutputDtype> initialized_output_dtype_;
};

} // namespace facebook::torchcodec
