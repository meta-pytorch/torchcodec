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

// Key for device interface registration with device type + variant support
struct DeviceInterfaceKey {
  torch::DeviceType deviceType;
  std::string_view variant = "default"; // e.g., "default", "beta", etc.

  bool operator<(const DeviceInterfaceKey& other) const {
    if (deviceType != other.deviceType) {
      return deviceType < other.deviceType;
    }
    return variant < other.variant;
  }

  explicit DeviceInterfaceKey(torch::DeviceType type) : deviceType(type) {}

  DeviceInterfaceKey(torch::DeviceType type, const std::string_view& variant)
      : deviceType(type), variant(variant) {}
};

class DeviceInterface {
 public:
  DeviceInterface(const torch::Device& device) : device_(device) {}

  virtual ~DeviceInterface(){};

  torch::Device& device() {
    return device_;
  };

  virtual std::optional<const AVCodec*> findCodec(
      [[maybe_unused]] const AVCodecID& codecId) {
    return std::nullopt;
  };

  // Initialize the device with parameters generic to all kinds of decoding.
  virtual void initialize(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx) = 0;

  // Initialize the device with parameters specific to video decoding. There is
  // a default empty implementation.
  virtual void initializeVideo(
      [[maybe_unused]] const VideoStreamOptions& videoStreamOptions,
      [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>&
          transforms,
      [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims) {}

  // In order for decoding to actually happen on an FFmpeg managed hardware
  // device, we need to register the DeviceInterface managed
  // AVHardwareDeviceContext with the AVCodecContext. We don't need to do this
  // on the CPU and if FFmpeg is not managing the hardware device.
  virtual void registerHardwareDeviceWithCodec(
      [[maybe_unused]] AVCodecContext* codecContext) {}

  virtual void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt) = 0;

  // ------------------------------------------
  // Extension points for custom decoding paths
  // ------------------------------------------

  // Set the codec context for default FFmpeg decoding operations
  // This must be called during initialization before using
  // sendPacket/receiveFrame
  virtual void setCodecContext(AVCodecContext* codecContext) {
    codecContext_ = codecContext;
  }

  // Returns AVSUCCESS on success, AVERROR(EAGAIN) if decoder queue full, or
  // other AVERROR on failure
  // Default implementation uses FFmpeg directly
  virtual int sendPacket(ReferenceAVPacket& avPacket) {
    if (!codecContext_) {
      TORCH_CHECK(
          false, "Codec context not available for default packet sending");
      return AVERROR(EINVAL);
    }
    return avcodec_send_packet(codecContext_, avPacket.get());
  }

  // Send an EOF packet to flush the decoder
  // Returns AVSUCCESS on success, or other AVERROR on failure
  // Default implementation uses FFmpeg directly
  virtual int sendEOFPacket() {
    if (!codecContext_) {
      TORCH_CHECK(false, "Codec context not available for EOF packet sending");
      return AVERROR(EINVAL);
    }
    return avcodec_send_packet(codecContext_, nullptr);
  }

  // Returns AVSUCCESS on success, AVERROR(EAGAIN) if no frame ready,
  // AVERROR_EOF if end of stream, or other AVERROR on failure
  // Default implementation uses FFmpeg directly
  virtual int receiveFrame(UniqueAVFrame& avFrame) {
    if (!codecContext_) {
      TORCH_CHECK(false, "Codec context not available for frame receiving");
      return AVERROR(EINVAL);
    }
    return avcodec_receive_frame(codecContext_, avFrame.get());
  }

  // Flush remaining frames from decoder
  virtual void flush() {
    // Default implementation is no-op for standard decoders
    // Custom decoders can override this method
  }

 protected:
  torch::Device device_;
  AVCodecContext* codecContext_ = nullptr; // Non-owning pointer
};

using CreateDeviceInterfaceFn =
    std::function<DeviceInterface*(const torch::Device& device)>;

bool registerDeviceInterface(
    const DeviceInterfaceKey& key,
    const CreateDeviceInterfaceFn createInterface);

void validateDeviceInterface(
    const std::string device,
    const std::string variant);

std::unique_ptr<DeviceInterface> createDeviceInterface(
    const torch::Device& device,
    const std::string_view variant = "default");

torch::Tensor rgbAVFrameToTensor(const UniqueAVFrame& avFrame);

} // namespace facebook::torchcodec
