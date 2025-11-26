// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include <memory>
#include <optional>

#include "CUDACommon.h"
#include "FFMPEGCommon.h"
#include "StreamOptions.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/buffer.h>
#include <libavutil/hwcontext.h>
}

namespace facebook::torchcodec {

class GpuEncoder {
 public:
  explicit GpuEncoder(const torch::Device& device);
  ~GpuEncoder();

  std::optional<const AVCodec*> findEncoder(const AVCodecID& codecId);
  void registerHardwareDeviceWithCodec(AVCodecContext* codecContext);
  void setupEncodingContext(AVCodecContext* codecContext);

  UniqueAVFrame convertTensorToAVFrame(
      const torch::Tensor& tensor,
      AVPixelFormat targetFormat,
      int frameIndex,
      AVCodecContext* codecContext);

  const torch::Device& device() const {
    return device_;
  }

 private:
  torch::Device device_;
  UniqueAVBufferRef hardwareDeviceCtx_;
  UniqueNppContext nppCtx_;

  void initializeHardwareContext();
  void setupHardwareFrameContext(AVCodecContext* codecContext);

  UniqueAVFrame convertRGBTensorToNV12Frame(
      const torch::Tensor& tensor,
      int frameIndex,
      AVCodecContext* codecContext);
};

} // namespace facebook::torchcodec
