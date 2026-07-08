// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <string_view>

#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

class FORCE_PUBLIC_VISIBILITY ColorConverter {
 public:
  explicit ColorConverter(
      const StableDevice& device = StableDevice(kStableCPU),
      std::string_view device_variant = "default");

  torch::stable::Tensor convert(UniqueAVFrame& av_frame);

 private:
  std::unique_ptr<DeviceInterface> device_interface_;
};

} // namespace facebook::torchcodec
