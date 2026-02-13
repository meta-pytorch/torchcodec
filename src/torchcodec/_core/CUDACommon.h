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

inline cudaStream_t getCurrentCudaStream(int32_t deviceIndex) {
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
