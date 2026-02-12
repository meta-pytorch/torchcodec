// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// See test_third_party_interface.py for context.
#include "DeviceInterface.h"
#include "FilterGraph.h"

namespace facebook::torchcodec {

class DummyDeviceInterface : public DeviceInterface {
 public:
  DummyDeviceInterface(const StableDevice& device) : DeviceInterface(device) {}

  virtual ~DummyDeviceInterface() {}

  void initialize(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx,
      const SharedAVCodecContext& codecContext) override {}

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override {}

 private:
  std::unique_ptr<FilterGraph> filterGraphContext_;
};

namespace {
static bool g_dummy = registerDeviceInterface(
    DeviceInterfaceKey(StableDeviceType::PrivateUse1),
    [](const StableDevice& device) {
      return new DummyDeviceInterface(device);
    });
} // namespace
} // namespace facebook::torchcodec
