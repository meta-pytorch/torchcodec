// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "CUDACommon.h"
#include "EncodingDeviceInterface.h"

namespace facebook::torchcodec {

// CUDA implementation of EncodingDeviceInterface.
// Uses NPP for RGB-to-NV12 color conversion on the GPU.
class EncodingCudaDeviceInterface : public EncodingDeviceInterface {
 public:
  explicit EncodingCudaDeviceInterface(const torch::Device& device);
  virtual ~EncodingCudaDeviceInterface();

  void initialize(AVCodecContext* codecContext) override;

  std::optional<const AVCodec*> findHardwareEncoder(
      const AVCodecID& codecId) override;

  void setupHardwareFrameContext(AVCodecContext* codecContext) override;

  void registerHardwareDeviceWithCodec(AVCodecContext* codecContext) override;

  UniqueAVFrame convertTensorToAVFrame(
      const torch::Tensor& tensor,
      int frameIndex,
      AVCodecContext* codecContext) override;

  AVPixelFormat getEncodingPixelFormat() const override {
    return CUDA_ENCODING_PIXEL_FORMAT;
  }

 private:
  const Npp32f (*getConversionMatrix(AVCodecContext* codecContext))[4];

  UniqueAVBufferRef hardwareDeviceCtx_;
  UniqueNppContext nppCtx_;
};

} // namespace facebook::torchcodec
