// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include <functional>
#include <memory>
#include <optional>
#include "FFMPEGCommon.h"

namespace facebook::torchcodec {

// Key for encoding device interface registration
struct EncodingDeviceInterfaceKey {
  torch::DeviceType deviceType;

  bool operator<(const EncodingDeviceInterfaceKey& other) const {
    return deviceType < other.deviceType;
  }

  explicit EncodingDeviceInterfaceKey(torch::DeviceType type)
      : deviceType(type) {}
};

// Pixel format used for encoding on CUDA devices (NV12)
constexpr AVPixelFormat CUDA_ENCODING_PIXEL_FORMAT = AV_PIX_FMT_NV12;

// Base class for device-specific encoding functionality.
// This interface abstracts the tensor-to-AVFrame conversion and hardware
// setup for video encoding across different devices (CPU, CUDA).
class EncodingDeviceInterface {
 public:
  explicit EncodingDeviceInterface(const torch::Device& device)
      : device_(device) {}

  virtual ~EncodingDeviceInterface() = default;

  torch::Device& device() {
    return device_;
  }

  // Initialize the encoding interface with codec context.
  // Called after codec context is configured but before encoding begins.
  virtual void initialize(AVCodecContext* codecContext) = 0;

  // Find a hardware encoder for the given codec ID.
  // Returns nullopt if no hardware encoder is available (e.g., CPU device).
  virtual std::optional<const AVCodec*> findHardwareEncoder(
      [[maybe_unused]] const AVCodecID& codecId) {
    return std::nullopt;
  }

  // Setup hardware frame context for encoding (CUDA-specific).
  // This allocates and initializes AVHWFramesContext needed by FFmpeg
  // to allocate frames on the GPU's memory.
  virtual void setupHardwareFrameContext(
      [[maybe_unused]] AVCodecContext* codecContext) {}

  // Register hardware device with codec (for hw-accelerated encoding).
  // Sets up the hw_device_ctx in the codec context.
  virtual void registerHardwareDeviceWithCodec(
      [[maybe_unused]] AVCodecContext* codecContext) {}

  // Convert a tensor to an AVFrame suitable for encoding.
  // Input tensor should be CHW format (3, H, W) with uint8 dtype.
  // Returns an AVFrame in the appropriate pixel format for the encoder.
  virtual UniqueAVFrame convertTensorToAVFrame(
      const torch::Tensor& tensor,
      int frameIndex,
      AVCodecContext* codecContext) = 0;

  // Get the pixel format used for encoding on this device.
  virtual AVPixelFormat getEncodingPixelFormat() const = 0;

 protected:
  torch::Device device_;
};

using CreateEncodingDeviceInterfaceFn =
    std::function<EncodingDeviceInterface*(const torch::Device& device)>;

// Register an encoding device interface for a specific device type.
bool registerEncodingDeviceInterface(
    const EncodingDeviceInterfaceKey& key,
    const CreateEncodingDeviceInterfaceFn createInterface);

// Create an encoding device interface for the specified device.
std::unique_ptr<EncodingDeviceInterface> createEncodingDeviceInterface(
    const torch::Device& device);

} // namespace facebook::torchcodec
