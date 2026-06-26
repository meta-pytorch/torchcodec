// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <functional>

namespace facebook::torchcodec {

// Torch-free seam for obtaining the "current" CUDA stream.
//
// getCurrentCudaStream() (declared in CUDACommon.h) returns the stream decode
// work should run on. The behavior depends on whether a provider was installed:
//   - torch present -> the torch adapter (TCTorchBackend.cpp) registers a
//     provider that returns torch's current CUDA stream for the given device
//     index, so decoded GPU frames stay synchronized with the user's torch
//     stream (this preserves the original behavior).
//   - torch absent  -> no provider is registered and the default CUDA stream
//     (stream 0) is used.
//
// The stream is passed as void* (rather than cudaStream_t) so that this header
// and the torch adapter that registers the provider do not need to include
// <cuda_runtime.h>. CUDACommon casts it back to cudaStream_t.
using CudaStreamProviderFn = std::function<void*(int32_t deviceIndex)>;

// Install (or, with nullptr, clear) the process-wide CUDA stream provider.
// Expected to be installed once at module init (not thread-safe to swap).
void setCudaStreamProvider(CudaStreamProviderFn fn);

} // namespace facebook::torchcodec
