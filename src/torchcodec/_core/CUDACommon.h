// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cuda_runtime.h>
#include <npp.h>

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

// NV12 and NPP require even dimensions. This rounds up to the nearest even
// value.
inline int roundUpToEven(int value) {
  return (value + 1) & ~1;
}

cudaStream_t getCurrentCudaStream(int32_t deviceIndex);

void initializeCudaContextWithPytorch(const StableDevice& device);

// Unique pointer type for NPP stream context
using UniqueNppContext = std::unique_ptr<NppStreamContext>;

// Convert an NV12 frame (on GPU) to an RGB tensor. The avFrame must have even
// width/height matching its actual NV12 data layout.
// outputDims is the desired output size. If smaller than the avFrame
// dimensions, the result is cropped. This is used when the original video has
// odd dimensions: the NV12 data is padded to even sizes, and outputDims
// carries the original (odd) size to crop back to.
torch::stable::Tensor convertNV12FrameToRGB(
    UniqueAVFrame& avFrame,
    const StableDevice& device,
    const UniqueNppContext& nppCtx,
    cudaStream_t nvdecStream,
    std::optional<torch::stable::Tensor> preAllocatedOutputTensor,
    const FrameDims& outputDims);

UniqueNppContext getNppStreamContext(const StableDevice& device);
void returnNppStreamContextToCache(
    const StableDevice& device,
    UniqueNppContext nppCtx);

void validatePreAllocatedTensorShape(
    const std::optional<torch::stable::Tensor>& preAllocatedOutputTensor,
    const FrameDims& frameDims);

int getDeviceIndex(const StableDevice& device);

} // namespace facebook::torchcodec
