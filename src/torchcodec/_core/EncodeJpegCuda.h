// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>

#include <cstdint>

#include "IOInterface.h"
#include "StableABICompat.h"

#if TORCHCODEC_ENABLE_NVJPEG
#include <cuda_runtime.h>
#include <nvjpeg.h>
#endif

namespace facebook::torchcodec {

FORCE_PUBLIC_VISIBILITY void encode_jpeg_cuda(
    const torch::stable::Tensor& image,
    int64_t quality,
    IOInterface& interface);

FORCE_PUBLIC_VISIBILITY torch::stable::Tensor encode_jpeg_to_tensor_cuda(
    const torch::stable::Tensor& image,
    int64_t quality);

#if TORCHCODEC_ENABLE_NVJPEG

class CUDAJpegEncoder {
 public:
  explicit CUDAJpegEncoder(const torch::stable::Device& target_device);
  ~CUDAJpegEncoder();

  torch::stable::Tensor encode_to_tensor(
      const torch::stable::Tensor& image,
      int64_t quality,
      cudaStream_t stream,
      const torch::stable::Device& output_device);

 private:
  const torch::stable::Device target_device_;

  nvjpegHandle_t nvjpeg_handle_;
  nvjpegEncoderState_t nvjpeg_enc_state_;
  nvjpegEncoderParams_t nvjpeg_enc_params_;
};

#endif // TORCHCODEC_ENABLE_NVJPEG

} // namespace facebook::torchcodec
