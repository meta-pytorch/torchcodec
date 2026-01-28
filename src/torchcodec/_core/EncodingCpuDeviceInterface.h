// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "EncodingDeviceInterface.h"
#include "FFMPEGCommon.h"

namespace facebook::torchcodec {

// CPU implementation of EncodingDeviceInterface.
// Uses libswscale for tensor-to-AVFrame conversion with color space conversion.
class EncodingCpuDeviceInterface : public EncodingDeviceInterface {
 public:
  explicit EncodingCpuDeviceInterface(const torch::Device& device);
  virtual ~EncodingCpuDeviceInterface() = default;

  void initialize(AVCodecContext* codecContext) override;

  UniqueAVFrame convertTensorToAVFrame(
      const torch::Tensor& tensor,
      int frameIndex,
      AVCodecContext* codecContext) override;

  AVPixelFormat getEncodingPixelFormat() const override;

 private:
  UniqueSwsContext swsContext_;

  int inWidth_ = -1;
  int inHeight_ = -1;
  AVPixelFormat inPixelFormat_ = AV_PIX_FMT_GBRP;

  int outWidth_ = -1;
  int outHeight_ = -1;
  AVPixelFormat outPixelFormat_ = AV_PIX_FMT_NONE;
};

} // namespace facebook::torchcodec
