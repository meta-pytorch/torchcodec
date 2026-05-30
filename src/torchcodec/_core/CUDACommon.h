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

void initializeCudaContextWithPytorch(const StableDevice& device);

void validatePreAllocatedTensorShape(
    const std::optional<torch::stable::Tensor>& preAllocatedOutputTensor,
    const FrameDims& frameDims);

int getDeviceIndex(const StableDevice& device);

} // namespace facebook::torchcodec
