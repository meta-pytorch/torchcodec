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

// ===========================================================================
// CUDA Stream Utilities
// ===========================================================================
// C++ wrapper for aoti_torch_get_current_cuda_stream.
// Returns the current CUDA stream for the given device index.
// This is the stable ABI way to get a cudaStream_t for use with CUDA libraries
// (NPP, NVDEC, cuBLAS, etc.).
//
// Note: USE_CUDA must be defined at compile time (set via CMake when ENABLE_CUDA is on).
inline cudaStream_t getCurrentCudaStream(int32_t deviceIndex) {
  void* stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(deviceIndex, &stream));
  return static_cast<cudaStream_t>(stream);
}

void initializeCudaContextWithPytorch(const StableDevice& device);

// Unique pointer type for NPP stream context
using UniqueNppContext = std::unique_ptr<NppStreamContext>;

StableTensor convertNV12FrameToRGB(
    UniqueAVFrame& avFrame,
    const StableDevice& device,
    const UniqueNppContext& nppCtx,
    cudaStream_t nvdecStream,
    std::optional<StableTensor> preAllocatedOutputTensor = std::nullopt);

UniqueNppContext getNppStreamContext(const StableDevice& device);
void returnNppStreamContextToCache(
    const StableDevice& device,
    UniqueNppContext nppCtx);

void validatePreAllocatedTensorShape(
    const std::optional<StableTensor>& preAllocatedOutputTensor,
    const UniqueAVFrame& avFrame);

int getDeviceIndex(const StableDevice& device);

} // namespace facebook::torchcodec
