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

cudaStream_t getCurrentCudaStream(int32_t deviceIndex) {
  // This is the documented and blessed way to get the current CUDA stream with
  // the stable ABI. aoti_torch_get_current_cuda_stream, TORCH_ERROR_CODE_CHECK,
  // and the corresponding torch/csrc/inductor/aoti_torch/c/shim.h header are
  // all safe to use:
  // https://github.com/pytorch/pytorch/blob/7bc8d4b0648e1d364dce0104c3aea2e7e3c1640a/docs/cpp/source/stable.rst?plain=1#L172-L179
  void* stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(deviceIndex, &stream));
  // Note: no need for checking against nullptr stream, it's a valid default
  // stream value.
  return static_cast<cudaStream_t>(stream);
}

// Make waitingStream wait until all work currently enqueued on runningStream
// has completed.
void syncStreams(cudaStream_t runningStream, cudaStream_t waitingStream) {
  cudaEvent_t event;
  cudaError_t err = cudaEventCreate(&event);
  STD_TORCH_CHECK(
      err == cudaSuccess, "cudaEventCreate failed: ", cudaGetErrorString(err));

  err = cudaEventRecord(event, runningStream);
  STD_TORCH_CHECK(
      err == cudaSuccess, "cudaEventRecord failed: ", cudaGetErrorString(err));

  err = cudaStreamWaitEvent(waitingStream, event, 0);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "cudaStreamWaitEvent failed: ",
      cudaGetErrorString(err));

  cudaEventDestroy(event);
}

void initializeCudaContextWithPytorch(const StableDevice& device) {
  // It is important for pytorch itself to create the cuda context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the cuda context.
  torch::stable::Tensor dummyTensorForCudaInitialization = torch::stable::empty(
      {1}, kStableUInt8, std::nullopt, StableDevice(device));
  torch::stable::zero_(dummyTensorForCudaInitialization);
}

void validatePreAllocatedTensorShape(
    const std::optional<torch::stable::Tensor>& preAllocatedOutputTensor,
    const FrameDims& frameDims) {
  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    STD_TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == frameDims.height) &&
            (shape[1] == frameDims.width) && (shape[2] == 3),
        "Expected tensor of shape ",
        frameDims.height,
        "x",
        frameDims.width,
        "x3, got ",
        intArrayRefToString(shape));
  }
}

int getDeviceIndex(const StableDevice& device) {
  // PyTorch uses int8_t as its torch::DeviceIndex, but FFmpeg and CUDA
  // libraries use int. So we use int, too.
  int deviceIndex = static_cast<int>(device.index());
  STD_TORCH_CHECK(
      deviceIndex >= -1 && deviceIndex < MAX_CUDA_GPUS,
      "Invalid device index = ",
      deviceIndex);

  if (deviceIndex == -1) {
    STD_TORCH_CHECK(
        cudaGetDevice(&deviceIndex) == cudaSuccess,
        "Failed to get current CUDA device.");
  }
  return deviceIndex;
}

} // namespace facebook::torchcodec
