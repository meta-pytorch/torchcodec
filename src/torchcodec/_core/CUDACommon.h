// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <npp.h>
#include <torch/types.h>

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

void initializeCudaContextWithPytorch(const torch::Device& device);

// Unique pointer type for NPP stream context
using UniqueNppContext = std::unique_ptr<NppStreamContext>;

torch::Tensor convertNV12FrameToRGB(
    UniqueAVFrame& avFrame,
    const torch::Device& device,
    const UniqueNppContext& nppCtx,
    at::cuda::CUDAStream nvdecStream,
    std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);

UniqueNppContext getNppStreamContext(const torch::Device& device);
void returnNppStreamContextToCache(
    const torch::Device& device,
    UniqueNppContext nppCtx);

void validatePreAllocatedTensorShape(
    const std::optional<torch::Tensor>& preAllocatedOutputTensor,
    const UniqueAVFrame& avFrame);

int getDeviceIndex(const torch::Device& device);

// Get or create a hardware device context for the given CUDA device.
// This is shared by both CudaDeviceInterface (decoding) and GpuEncoder
// (encoding).
UniqueAVBufferRef getHardwareDeviceContext(const torch::Device& device);

// Add a hardware device context to the cache for reuse.
void addHardwareDeviceContextToCache(
    const torch::Device& device,
    UniqueAVBufferRef hardwareDeviceCtx);

} // namespace facebook::torchcodec
