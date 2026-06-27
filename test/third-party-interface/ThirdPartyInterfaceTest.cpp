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
  DummyDeviceInterface(const tc::Device& device) : DeviceInterface(device) {}

  virtual ~DummyDeviceInterface() {}

  void initialize(const SharedAVCodecContext& codecContext) override {
    // Access to CPU device interface is essential for CPU fallback
    // implementation
    std::unique_ptr<DeviceInterface> cpuInterface =
        createDeviceInterface(tc::Device(tc::kCPU));
  }

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<tc::Tensor> preAllocatedOutputTensor =
          std::nullopt) override {}

 private:
  std::unique_ptr<FilterGraph> filterGraphContext_;
};

namespace {
static bool g_dummy = registerDeviceInterface(
    DeviceInterfaceKey(tc::kPrivateUse1),
    [](const tc::Device& device) { return new DummyDeviceInterface(device); });
} // namespace
} // namespace facebook::torchcodec
