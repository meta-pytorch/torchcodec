// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>

#include <cstdint>
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

// Encodes a single CHW uint8 CUDA image tensor into a JPEG and returns the
// bitstream as a 1-D uint8 tensor on the *same CUDA device* as the input. This
// is zero-copy (the bytes never leave the GPU); callers wanting host bytes can
// .cpu() the result.
FORCE_PUBLIC_VISIBILITY torch::stable::Tensor encode_jpeg_to_tensor_cuda(
    const torch::stable::Tensor& image,
    int64_t quality);

#if TORCHCODEC_ENABLE_NVJPEG

class CUDAJpegEncoder {
 public:
  explicit CUDAJpegEncoder(const torch::stable::Device& target_device);
  ~CUDAJpegEncoder();

  // Encodes `image` and returns the JPEG bitstream as host (CPU) bytes.
<<<<<<< Updated upstream
  std::vector<uint8_t> encode_image(
||||||| Stash base
  std::vector<uint8_t> encode_image(
=======
  std::vector<uint8_t> encode_to_host_vector(
      const torch::stable::Tensor& image,
      int64_t quality,
      cudaStream_t stream);

  // Encodes `image` and returns the JPEG bitstream as a 1-D uint8 tensor on the
  // encoder's CUDA device (no device-to-host copy).
  torch::stable::Tensor encode_to_device_tensor(
>>>>>>> Stashed changes
      const torch::stable::Tensor& image,
      int64_t quality,
      cudaStream_t stream);

 private:
  // Runs nvjpegEncodeImage for `image` and returns the encoded bitstream length,
  // leaving the bitstream in nvjpeg_enc_state_ for retrieval.
  size_t encode_and_get_length(
      const torch::stable::Tensor& image,
      int64_t quality,
      cudaStream_t stream);

  const torch::stable::Device target_device_;

  nvjpegHandle_t nvjpeg_handle_;
  nvjpegEncoderState_t nvjpeg_enc_state_;
  nvjpegEncoderParams_t nvjpeg_enc_params_;
};

#endif // TORCHCODEC_ENABLE_NVJPEG

} // namespace facebook::torchcodec
