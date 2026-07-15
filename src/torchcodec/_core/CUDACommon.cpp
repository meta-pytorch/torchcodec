// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "CUDACommon.h"
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include "StableABICompat.h"
#include "ValidationUtils.h"

namespace facebook::torchcodec {

cudaStream_t get_current_cuda_stream(int32_t device_index) {
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
void sync_streams(cudaStream_t running_stream, cudaStream_t waiting_stream) {
  cudaEvent_t event;
  cudaError_t err = cudaEventCreate(&event);
  STD_TORCH_CHECK(
      err == cudaSuccess, "cudaEventCreate failed: ", cudaGetErrorString(err));

  err = cudaEventRecord(event, running_stream);
  STD_TORCH_CHECK(
      err == cudaSuccess, "cudaEventRecord failed: ", cudaGetErrorString(err));

  err = cudaStreamWaitEvent(waiting_stream, event, 0);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "cudaStreamWaitEvent failed: ",
      cudaGetErrorString(err));

  cudaEventDestroy(event);
}

void initialize_cuda_context_with_pytorch(const StableDevice& device) {
  // It is important for pytorch itself to create the cuda context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the cuda context.
  torch::stable::Tensor dummy_tensor_for_cuda_initialization =
      torch::stable::empty(
          {1}, kStableUInt8, std::nullopt, StableDevice(device));
  torch::stable::zero_(dummy_tensor_for_cuda_initialization);
}

void validate_pre_allocated_tensor_shape(
    const std::optional<torch::stable::Tensor>& pre_allocated_output_tensor,
    const FrameDims& frame_dims) {
  if (pre_allocated_output_tensor.has_value()) {
    auto shape = pre_allocated_output_tensor.value().sizes();
    STD_TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == frame_dims.height) &&
            (shape[1] == frame_dims.width) && (shape[2] == 3),
        "Expected tensor of shape ",
        frame_dims.height,
        "x",
        frame_dims.width,
        "x3, got ",
        int_array_ref_to_string(shape));
  }
}

int get_device_index(const StableDevice& device) {
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

} // namespace facebook::torchcodec
