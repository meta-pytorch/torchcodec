// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include "CUDACommon.h"
#include "FFMPEGCommon.h"

namespace facebook::torchcodec {

// GpuEncoder handles hardware-accelerated video encoding on CUDA devices.
// This class isolates CUDA encoding dependencies from the DeviceInterface
// hierarchy, keeping DeviceInterfaces focused on decoding only.
class GpuEncoder {
 public:
  explicit GpuEncoder(const torch::Device& device);
  ~GpuEncoder();

  // Find a hardware encoder for the given codec ID.
  // Returns the codec if found, std::nullopt otherwise.
  std::optional<const AVCodec*> findHardwareEncoder(const AVCodecID& codecId);

  // Register the hardware device context with the codec context for encoding.
  void registerHardwareDeviceWithCodec(AVCodecContext* codecContext);

  // Allocate and initialize AVHWFramesContext for encoding.
  // Sets pixel format fields to enable encoding with CUDA device.
  void setupHardwareFrameContextForEncoding(AVCodecContext* codecContext);

  // Convert a CUDA tensor to an AVFrame suitable for encoding.
  UniqueAVFrame convertCUDATensorToAVFrameForEncoding(
      const torch::Tensor& tensor,
      int frameIndex,
      AVCodecContext* codecContext);

  // Pixel format used for encoding on CUDA devices
  static constexpr AVPixelFormat CUDA_ENCODING_PIXEL_FORMAT = AV_PIX_FMT_NV12;

 private:
  torch::Device device_;
  UniqueAVBufferRef hardwareDeviceCtx_;
  UniqueNppContext nppCtx_;
};

// Factory function to create a GpuEncoder instance.
std::unique_ptr<GpuEncoder> createGpuEncoder(const torch::Device& device);

} // namespace facebook::torchcodec
