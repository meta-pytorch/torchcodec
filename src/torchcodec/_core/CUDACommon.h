// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/Exception.h>

#include <optional>

#include "StableABICompat.h"

namespace facebook::torchcodec {

struct FrameDims;

// Pytorch can only handle up to 128 GPUs.
// https://github.com/pytorch/pytorch/blob/e30c55ee527b40d67555464b9e402b4b7ce03737/c10/cuda/CUDAMacros.h#L44
constexpr int MAX_CUDA_GPUS = 128;

// NV12 requires even dimensions. This rounds up to the nearest even value.
inline int round_up_to_even(int value) {
  return (value + 1) & ~1;
}

// Defined inline (rather than in CUDACommon.cpp) so libtorchcodec_image, which
// does not link the core library, can use it too.
inline int get_device_index(const StableDevice& device) {
  // PyTorch uses int8_t as its torch::DeviceIndex, but FFmpeg and CUDA
  // libraries use int. So we use int, too.
  int device_index = static_cast<int>(device.index());
  STD_TORCH_CHECK(
      device_index >= -1 && device_index < MAX_CUDA_GPUS,
      "Invalid device index = ",
      device_index);

  if (device_index == -1) {
    STD_TORCH_CHECK(
        cudaGetDevice(&device_index) == cudaSuccess,
        "Failed to get current CUDA device.");
  }
  return device_index;
}

// Defined inline (rather than in CUDACommon.cpp) so libtorchcodec_image, which
// does not link the core library, can use it too.
inline cudaStream_t get_current_cuda_stream(int32_t device_index) {
  // This is the documented and blessed way to get the current CUDA stream with
  // the stable ABI. aoti_torch_get_current_cuda_stream, TORCH_ERROR_CODE_CHECK,
  // and the corresponding torch/csrc/inductor/aoti_torch/c/shim.h header are
  // all safe to use:
  // https://github.com/pytorch/pytorch/blob/7bc8d4b0648e1d364dce0104c3aea2e7e3c1640a/docs/cpp/source/stable.rst?plain=1#L172-L179
  void* stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream));
  // Note: no need for checking against nullptr stream, it's a valid default
  // stream value.
  return static_cast<cudaStream_t>(stream);
}

// Make waitingStream wait until all work currently enqueued on runningStream
// has completed.
void sync_streams(cudaStream_t running_stream, cudaStream_t waiting_stream);

void initialize_cuda_context_with_pytorch(const StableDevice& device);

void validate_pre_allocated_tensor_shape(
    const std::optional<torch::stable::Tensor>& pre_allocated_output_tensor,
    const FrameDims& frame_dims);

} // namespace facebook::torchcodec
