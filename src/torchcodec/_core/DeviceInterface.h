// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include "FFMPEGCommon.h"
#include "Frame.h"
#include "StableABICompat.h"
#include "StreamOptions.h"
#include "Transform.h"

namespace facebook::torchcodec {

// Key for device interface registration with device type + variant support
struct DeviceInterfaceKey {
  StableDeviceType deviceType;
  std::string_view variant = "default"; // e.g., "default", "ffmpeg"

  bool operator<(const DeviceInterfaceKey& other) const {
    if (deviceType != other.deviceType) {
      return deviceType < other.deviceType;
    }
    return variant < other.variant;
  }

  explicit DeviceInterfaceKey(StableDeviceType type) : deviceType(type) {}

  DeviceInterfaceKey(StableDeviceType type, const std::string_view& variant)
      : deviceType(type), variant(variant) {}
};

class DeviceInterface {
 public:
  DeviceInterface(const StableDevice& device) : device_(device) {}

  virtual ~DeviceInterface(){};

  StableDevice& device() {
    return device_;
  };

  virtual std::optional<const AVCodec*> findCodec(
      [[maybe_unused]] const AVCodecID& codecId,
      [[maybe_unused]] bool isDecoder = true) {
    return std::nullopt;
  };

  // Initialize the device with the codec context, which is needed for the
  // default sendPacket/receiveFrame/flush implementations.
  virtual void initialize(const SharedAVCodecContext& codecContext) = 0;

  // Initialize the device with parameters specific to video decoding. There is
  // a default empty implementation.
  virtual void initializeVideo(
      [[maybe_unused]] const AVStream* avStream,
      [[maybe_unused]] const UniqueDecodingAVFormatContext& avFormatCtx,
      [[maybe_unused]] const VideoStreamOptions& videoStreamOptions,
      [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>&
          transforms,
      [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims) {}

  // Initialize the device with parameters specific to audio decoding. There is
  // a default empty implementation.
  virtual void initializeAudio(
      [[maybe_unused]] const AudioStreamOptions& audioStreamOptions) {}

  // Flush any remaining samples from the audio resampler buffer.
  // When sample rate conversion is involved, some samples may be buffered
  // between frames for proper interpolation. This function flushes those
  // buffered samples.
  // Returns an optional tensor containing the flushed samples, or std::nullopt
  // if there are no buffered samples or audio is not supported.
  virtual std::optional<torch::stable::Tensor> maybeFlushAudioBuffers() {
    return std::nullopt;
  }

  // In order for decoding to actually happen on an FFmpeg managed hardware
  // device, we need to register the DeviceInterface managed
  // AVHardwareDeviceContext with the AVCodecContext. We don't need to do this
  // on the CPU and if FFmpeg is not managing the hardware device.
  virtual void registerHardwareDeviceWithCodec(
      [[maybe_unused]] AVCodecContext* codecContext) {}

  // The dtype that should be used for the pre-allocated tensors on batch APIs.
  // Usually that will just be the user's requested dtype, but not always: on
  // CUDA with SDR videos when float32 is asked, will fallback from P016 to NV12
  // to avoid falling back all the way to the CPU. The pre-allocated tensors
  // then need to be uint8, not float32.
  virtual OutputDtype getPreAllocationDtype(OutputDtype requestedDtype) const {
    return requestedDtype;
  }

  virtual void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::stable::Tensor> preAllocatedOutputTensor =
          std::nullopt) = 0;

  // ------------------------------------------
  // Extension points for custom decoding paths
  // ------------------------------------------

  // Returns AVSUCCESS on success, AVERROR(EAGAIN) if decoder queue full, or
  // other AVERROR on failure
  // Default implementation uses FFmpeg directly
  virtual int sendPacket(ReferenceAVPacket& avPacket) {
    STD_TORCH_CHECK(
        codecContext_ != nullptr,
        "Codec context not available for default packet sending");
    return avcodec_send_packet(codecContext_.get(), avPacket.get());
  }

  // Send an EOF packet to flush the decoder
  // Returns AVSUCCESS on success, or other AVERROR on failure
  // Default implementation uses FFmpeg directly
  virtual int sendEOFPacket() {
    STD_TORCH_CHECK(
        codecContext_ != nullptr,
        "Codec context not available for default EOF packet sending");
    return avcodec_send_packet(codecContext_.get(), nullptr);
  }

  // Returns AVSUCCESS on success, AVERROR(EAGAIN) if no frame ready,
  // AVERROR_EOF if end of stream, or other AVERROR on failure
  // Default implementation uses FFmpeg directly
  virtual int receiveFrame(UniqueAVFrame& avFrame) {
    STD_TORCH_CHECK(
        codecContext_ != nullptr,
        "Codec context not available for default frame receiving");
    return avcodec_receive_frame(codecContext_.get(), avFrame.get());
  }

  // Flush remaining frames from decoder
  virtual void flush() {
    STD_TORCH_CHECK(
        codecContext_ != nullptr,
        "Codec context not available for default flushing");
    avcodec_flush_buffers(codecContext_.get());

    // We also manually flush any remaining frames in the decoder buffer. We
    // shouldn't have to do this, because avcodec_flush_buffers should handle
    // it, but some codecs like HEVC may still have frames buffered internally
    // in edge cases (ex. hitting EOF) as observed in
    // https://github.com/meta-pytorch/torchcodec/issues/1339.
    UniqueAVFrame tmp(av_frame_alloc());
    while (avcodec_receive_frame(codecContext_.get(), tmp.get()) == AVSUCCESS) {
    }
  }

  virtual std::string getDetails() {
    return "";
  }

  // Pixel format used for encoding on CUDA devices
  static constexpr AVPixelFormat CUDA_ENCODING_PIXEL_FORMAT = AV_PIX_FMT_NV12;

  virtual UniqueAVFrame convertTensorToAVFrameForEncoding(
      [[maybe_unused]] const torch::stable::Tensor& tensor,
      [[maybe_unused]] int frameIndex,
      [[maybe_unused]] AVCodecContext* codecContext) {
    STD_TORCH_CHECK(false, "convertTensorToAVFrameForEncoding not implemented");
  }

  // Function used for video encoding, only implemented in CudaDeviceInterface.
  // It is here to isolate CUDA dependencies from CPU builds
  virtual void setupHardwareFrameContextForEncoding(
      [[maybe_unused]] AVCodecContext* codecContext) {
    STD_TORCH_CHECK(
        false, "setupHardwareFrameContextForEncoding not implemented");
  }

  virtual std::optional<const AVCodec*> findHardwareEncoder(
      [[maybe_unused]] const AVCodecID& codecId) {
    STD_TORCH_CHECK(false, "findHardwareEncoder not implemented");
  }

 protected:
  StableDevice device_;
  SharedAVCodecContext codecContext_;
  AVMediaType avMediaType_;
};

using CreateDeviceInterfaceFn =
    std::function<DeviceInterface*(const StableDevice& device)>;

TORCHCODEC_THIRD_PARTY_API bool registerDeviceInterface(
    const DeviceInterfaceKey& key,
    const CreateDeviceInterfaceFn createInterface);

FORCE_PUBLIC_VISIBILITY void validateDeviceInterface(
    const std::string& device,
    const std::string& variant);

TORCHCODEC_THIRD_PARTY_API std::unique_ptr<DeviceInterface>
createDeviceInterface(
    const StableDevice& device,
    const std::string_view variant = "default");

torch::stable::Tensor rgbAVFrameToTensor(const UniqueAVFrame& avFrame);

} // namespace facebook::torchcodec
