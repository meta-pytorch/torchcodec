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

  void initialize(const SharedAVCodecContext& codec_context) override {
    // Access to CPU device interface is essential for CPU fallback
    // implementation
    std::unique_ptr<DeviceInterface> cpu_interface =
        create_device_interface(kStableCPU);
  }

  void convert_av_frame_to_frame_output(
      UniqueAVFrame& av_frame,
      FrameOutput& frame_output,
      std::optional<torch::stable::Tensor> pre_allocated_output_tensor =
          std::nullopt) override {}

  // Encoder-side hooks added to exercise the third-party API surface at build
  // time.
  AVPixelFormat get_encoding_pixel_format(
      const AVCodec& /*av_codec*/,
      const std::optional<std::string>& /*user_pixel_format*/) const override {
    return AV_PIX_FMT_NONE;
  }

  void setup_hardware_frame_context_for_encoding(
      AVCodecContext* /*codec_context*/) override {}

 private:
  std::unique_ptr<FilterGraph> filter_graph_context_;
};

namespace {
static bool g_dummy = register_device_interface(
    DeviceInterfaceKey(StableDeviceType::PrivateUse1),
    [](const StableDevice& device) {
      return new DummyDeviceInterface(device);
    });
} // namespace
} // namespace facebook::torchcodec
