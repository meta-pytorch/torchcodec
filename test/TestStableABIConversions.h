// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// Test-only conversion utilities between StableTensor and at::Tensor.
//
// The stable ABI's AtenTensorHandle is documented as being at::Tensor* under
// the hood (see torch/csrc/inductor/aoti_torch/c/macros.h). We use this fact
// to provide conversion between the two types for testing purposes.
//
// IMPORTANT: These conversions are for testing ONLY and rely on implementation
// details that could change. The actual library code should use stable ABI
// types consistently.

// First, include standard PyTorch headers while TORCH_TARGET_VERSION is NOT defined.
// This gives us access to at::Tensor and the standard PyTorch APIs.
#include <ATen/Tensor.h>
#include <torch/torch.h>

// Now include the stable ABI headers from the library.
// We need to temporarily define TORCH_TARGET_VERSION for these headers.
#ifndef TORCH_TARGET_VERSION
#define TORCH_TARGET_VERSION 0x020b000000000000
#define TORCHCODEC_DEFINED_TORCH_TARGET_VERSION
#endif

#include "src/torchcodec/_core/StableABICompat.h"

#ifdef TORCHCODEC_DEFINED_TORCH_TARGET_VERSION
#undef TORCH_TARGET_VERSION
#undef TORCHCODEC_DEFINED_TORCH_TARGET_VERSION
#endif

namespace facebook::torchcodec {
namespace test_utils {

// Convert a StableTensor to an at::Tensor reference.
// This provides a view of the same underlying data.
// The returned reference is valid as long as the StableTensor is alive.
inline at::Tensor& toAtTensor(const StableTensor& stable) {
  // AtenTensorHandle is at::Tensor* under the hood
  return *reinterpret_cast<at::Tensor*>(stable.get());
}

// Convert an at::Tensor to a StableTensor.
// This creates a StableTensor that wraps the same underlying tensor.
// The at::Tensor must remain valid for the lifetime of the StableTensor.
inline StableTensor toStableTensor(const at::Tensor& tensor) {
  // We need to create a new AtenTensorHandle from the tensor.
  // Since AtenTensorHandle is at::Tensor*, we create a new at::Tensor on the
  // heap that shares storage with the input tensor.
  at::Tensor* newTensor = new at::Tensor(tensor);
  AtenTensorHandle handle = reinterpret_cast<AtenTensorHandle>(newTensor);
  // StableTensor constructor takes ownership of the handle
  return StableTensor(handle);
}

} // namespace test_utils
} // namespace facebook::torchcodec
