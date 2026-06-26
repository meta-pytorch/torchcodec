// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cuda_runtime.h>

#include "FFMPEGCommon.h"
#include "Frame.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

// Pytorch can only handle up to 128 GPUs.
// https://github.com/pytorch/pytorch/blob/e30c55ee527b40d67555464b9e402b4b7ce03737/c10/cuda/CUDAMacros.h#L44
constexpr int MAX_CUDA_GPUS = 128;

// NV12 requires even dimensions. This rounds up to the nearest even value.
inline int roundUpToEven(int value) {
  return (value + 1) & ~1;
}

cudaStream_t getCurrentCudaStream(int32_t deviceIndex);

// Make waitingStream wait until all work currently enqueued on runningStream
// has completed.
void syncStreams(cudaStream_t runningStream, cudaStream_t waitingStream);

// Ensure a CUDA context exists on `device`. This allocates a tiny tc::Tensor,
// which routes through tc's allocator hook: when torch is present that means
// torch creates/owns the (primary) context, so it stays compatible with torch;
// when torch is absent a plain cudaMalloc creates the primary context. Either
// way FFmpeg then reuses that context rather than creating an incompatible one.
void initializeCudaContext(const tc::Device& device);

// RAII guard that sets the current CUDA device for its lifetime and restores the
// previously-current device on destruction. Torch-free replacement for
// torch::stable::accelerator::DeviceGuard. A negative index is a no-op.
class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(int deviceIndex);
  ~CudaDeviceGuard();
  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

 private:
  int prevDeviceIndex_ = -1;
};

void validatePreAllocatedTensorShape(
    const std::optional<tc::Tensor>& preAllocatedOutputTensor,
    const FrameDims& frameDims);

int getDeviceIndex(const tc::Device& device);

} // namespace facebook::torchcodec
