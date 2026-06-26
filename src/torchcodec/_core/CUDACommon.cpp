// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "CUDACommon.h"
#include "CUDAStreamHook.h"
#include "TCError.h"
#include "ValidationUtils.h"

namespace facebook::torchcodec {

namespace {
// Process-wide stream provider; see CUDAStreamHook.h. When torch is present the
// torch adapter installs a provider returning torch's current stream; when
// absent this stays null and getCurrentCudaStream() returns the default stream.
CudaStreamProviderFn g_cudaStreamProvider = nullptr;
} // namespace

void setCudaStreamProvider(CudaStreamProviderFn fn) {
  g_cudaStreamProvider = std::move(fn);
}

cudaStream_t getCurrentCudaStream(int32_t deviceIndex) {
  if (g_cudaStreamProvider) {
    // torch present: return torch's current CUDA stream for this device, so
    // decoded GPU frames stay synchronized with the user's torch stream.
    return static_cast<cudaStream_t>(g_cudaStreamProvider(deviceIndex));
  }
  // torch absent: use the default (legacy) CUDA stream. This is a valid stream
  // value; nullptr / 0 denotes the default stream.
  return static_cast<cudaStream_t>(0);
}

// Make waitingStream wait until all work currently enqueued on runningStream
// has completed.
void syncStreams(cudaStream_t runningStream, cudaStream_t waitingStream) {
  cudaEvent_t event;
  cudaError_t err = cudaEventCreate(&event);
  TC_CHECK(
      err == cudaSuccess, "cudaEventCreate failed: ", cudaGetErrorString(err));

  err = cudaEventRecord(event, runningStream);
  TC_CHECK(
      err == cudaSuccess, "cudaEventRecord failed: ", cudaGetErrorString(err));

  err = cudaStreamWaitEvent(waitingStream, event, 0);
  TC_CHECK(
      err == cudaSuccess,
      "cudaStreamWaitEvent failed: ",
      cudaGetErrorString(err));

  cudaEventDestroy(event);
}

void initializeCudaContext(const tc::Device& device) {
  // It is important for the allocator (torch when present, else cudaMalloc) to
  // create the cuda context. If ffmpeg creates the context it may not be
  // compatible. This is a dummy tensor to initialize the cuda context; its
  // allocation goes through tc's allocator hook.
  tc::Tensor dummyTensorForCudaInitialization =
      tc::empty({1}, tc::kUInt8, std::nullopt, tc::Device(device));
  tc::zero_(dummyTensorForCudaInitialization);
}

CudaDeviceGuard::CudaDeviceGuard(int deviceIndex) {
  if (deviceIndex < 0) {
    return;
  }
  TC_CHECK(
      cudaGetDevice(&prevDeviceIndex_) == cudaSuccess,
      "Failed to get current CUDA device.");
  if (prevDeviceIndex_ != deviceIndex) {
    TC_CHECK(
        cudaSetDevice(deviceIndex) == cudaSuccess,
        "Failed to set CUDA device to ",
        deviceIndex);
  } else {
    // No switch needed; mark as no-op so the destructor doesn't restore.
    prevDeviceIndex_ = -1;
  }
}

CudaDeviceGuard::~CudaDeviceGuard() {
  if (prevDeviceIndex_ >= 0) {
    cudaSetDevice(prevDeviceIndex_);
  }
}

void validatePreAllocatedTensorShape(
    const std::optional<tc::Tensor>& preAllocatedOutputTensor,
    const FrameDims& frameDims) {
  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    TC_CHECK(
        (shape.size() == 3) && (shape[0] == frameDims.height) &&
            (shape[1] == frameDims.width) && (shape[2] == 3),
        "Expected tensor of shape ",
        frameDims.height,
        "x",
        frameDims.width,
        "x3, got ",
        tc::intArrayRefToString(shape));
  }
}

int getDeviceIndex(const tc::Device& device) {
  // PyTorch uses int8_t as its torch::DeviceIndex, but FFmpeg and CUDA
  // libraries use int. So we use int, too.
  int deviceIndex = static_cast<int>(device.index());
  TC_CHECK(
      deviceIndex >= -1 && deviceIndex < MAX_CUDA_GPUS,
      "Invalid device index = ",
      deviceIndex);

  if (deviceIndex == -1) {
    TC_CHECK(
        cudaGetDevice(&deviceIndex) == cudaSuccess,
        "Failed to get current CUDA device.");
  }
  return deviceIndex;
}

} // namespace facebook::torchcodec
