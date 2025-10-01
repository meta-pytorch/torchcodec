// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include "FFMPEGCommon.h"
#include "src/torchcodec/_core/Frame.h"
#include "src/torchcodec/_core/StreamOptions.h"
#include "src/torchcodec/_core/Transform.h"

namespace facebook::torchcodec {

class DeviceInterface {
 public:
  DeviceInterface(const torch::Device& device) : device_(device) {}

  virtual ~DeviceInterface(){};

  torch::Device& device() {
    return device_;
  };

  virtual std::optional<const AVCodec*> findCodec(const AVCodecID& codecId) = 0;

  // Initialize the device with parameters generic to all kinds of decoding.
  virtual void initialize(
      AVCodecContext* codecContext,
      const AVRational& timeBase) = 0;

  // Initialize the device with parameters specific to video decoding. There is
  // a default empty implementation.
  virtual void initializeVideo(
      [[maybe_unused]] const VideoStreamOptions& videoStreamOptions,
      [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>&
          transforms,
      [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims) {}

  virtual void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt) = 0;

 protected:
  torch::Device device_;
};

using CreateDeviceInterfaceFn =
    std::function<DeviceInterface*(const torch::Device& device)>;

bool registerDeviceInterface(
    torch::DeviceType deviceType,
    const CreateDeviceInterfaceFn createInterface);

torch::Device createTorchDevice(const std::string device);

std::unique_ptr<DeviceInterface> createDeviceInterface(
    const torch::Device& device);

torch::Tensor rgbAVFrameToTensor(const UniqueAVFrame& avFrame);

} // namespace facebook::torchcodec
