// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "IOInterface.h"
#include "StableABICompat.h"

#if TORCHCODEC_ENABLE_NVJPEG
#include <cuda_runtime.h>
#include <nvjpeg.h>
#endif

namespace facebook::torchcodec {

// Encodes a single CHW uint8 CUDA image tensor into a JPEG, writing the encoded
// bytes to `interface` (a file or file-like), mirroring the CPU encode_jpeg.
FORCE_PUBLIC_VISIBILITY void encode_jpeg_cuda(
    const torch::stable::Tensor& image,
    int64_t quality,
    IOInterface& interface);

#if TORCHCODEC_ENABLE_NVJPEG

class CUDAJpegEncoder {
 public:
  explicit CUDAJpegEncoder(const torch::stable::Device& target_device);
  ~CUDAJpegEncoder();

  // Encodes `image` and returns the JPEG bitstream as host (CPU) bytes.
  std::vector<uint8_t> encode_image(
      const torch::stable::Tensor& image,
      int64_t quality,
      cudaStream_t stream);

 private:
  const torch::stable::Device target_device_;

  nvjpegHandle_t nvjpeg_handle_;
  nvjpegEncoderState_t nvjpeg_enc_state_;
  nvjpegEncoderParams_t nvjpeg_enc_params_;
};

// A per-device pool of reusable CUDAJpegEncoder objects. Modeled on NVJpegCache
// (the JPEG decoder's cache): encoders are expensive to create, so we reuse
// them across calls.
class NVJpegEncoderCache {
 public:
  static NVJpegEncoderCache& get_cache(const torch::stable::Device& device);

  std::unique_ptr<CUDAJpegEncoder> get_encoder(
      const torch::stable::Device& device);

  void return_encoder(std::unique_ptr<CUDAJpegEncoder> encoder);

 private:
  static NVJpegEncoderCache* get_cache_instances();

  std::vector<std::unique_ptr<CUDAJpegEncoder>> pool_;
  std::mutex pool_lock_;
};

#endif // TORCHCODEC_ENABLE_NVJPEG

} // namespace facebook::torchcodec
