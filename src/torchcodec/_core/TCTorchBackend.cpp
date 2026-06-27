// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Wires tc::Tensor's pluggable hooks to PyTorch. This file is compiled into the
// torch-linked custom-ops library, so its static initializer runs when that
// library is loaded (torch.ops.load_library at import, or when a C++ test links
// it). The torch-free pybind frontend does NOT load this, so in a torch-free
// process the hooks stay unset (CPU malloc only; GPU unavailable) by design.
//
// - Allocator: allocate storage via torch (torch's caching allocator on CUDA),
//   so when torch is present GPU memory comes from its pool, not raw
//   cudaMalloc.
// - CUDA compute backend: run each non-CPU tc op through torch (toStable ->
//   torch op -> fromStable), zero-copy at the boundary.

#include <memory>
#include <vector>

#include "StableABICompat.h"
#include "TCStableConvert.h"
#include "TCTensor.h"

#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include "CUDAStreamHook.h"
#endif

namespace facebook::torchcodec {
namespace {

// Allocate `numBytes` of storage on `device` via torch, returning storage that
// owns the underlying torch tensor (released back to torch's allocator when the
// last tc reference dies).
std::shared_ptr<void>
allocViaTorch(int64_t numBytes, tc::ScalarType /*dtype*/, tc::Device device) {
  if (numBytes == 0) {
    return std::shared_ptr<void>(reinterpret_cast<void*>(1), [](void*) {});
  }
  // Allocate a 1-D byte buffer; tc::Tensor applies the real dtype/shape on top.
  torch::stable::Tensor storageTensor = torch::stable::empty(
      {numBytes}, kStableUInt8, std::nullopt, toStableDevice(device));
  void* dataPtr = storageTensor.mutable_data_ptr();
  return std::shared_ptr<void>(
      dataPtr, [storageTensor](void*) mutable { /* releases torch storage */ });
}

tc::DeviceBackend makeTorchCudaBackend() {
  tc::DeviceBackend backend;

  backend.copy_ = [](tc::Tensor& dst, const tc::Tensor& src) {
    // toStable(dst) shares dst's storage, so the in-place copy_ writes through.
    torch::stable::Tensor stableDst = toStable(dst);
    torch::stable::copy_(stableDst, toStable(src));
  };

  backend.zero_ = [](tc::Tensor& self) {
    torch::stable::Tensor stableSelf = toStable(self);
    torch::stable::zero_(stableSelf);
  };

  backend.toDtype = [](const tc::Tensor& self, tc::ScalarType dtype) {
    return fromStable(torch::stable::to(toStable(self), toStableDtype(dtype)));
  };

  backend.div = [](const tc::Tensor& self, double other) {
    return fromStable(stableDiv(toStable(self), other));
  };

  backend.contiguous = [](const tc::Tensor& self) {
    return fromStable(torch::stable::contiguous(toStable(self)));
  };

  backend.cat = [](const std::vector<tc::Tensor>& tensors, int64_t dim) {
    std::vector<torch::stable::Tensor> stableTensors;
    stableTensors.reserve(tensors.size());
    for (const auto& tensor : tensors) {
      stableTensors.push_back(toStable(tensor));
    }
    return fromStable(stableCat(stableTensors, dim));
  };

  backend.rot90 =
      [](const tc::Tensor& self, int64_t k, int64_t dim0, int64_t dim1) {
        return fromStable(
            stableRot90(toStable(self), static_cast<int>(k), dim0, dim1));
      };

  return backend;
}

// Registers the hooks at library load.
struct TorchHookRegistrar {
  TorchHookRegistrar() {
    tc::setAllocator(allocViaTorch);
    tc::registerDeviceBackend(tc::DeviceType::CUDA, makeTorchCudaBackend());
#ifdef USE_CUDA
    // Return torch's current CUDA stream so decoded GPU frames stay
    // synchronized with the user's torch stream (the original behavior). Passed
    // as void* to keep CUDAStreamHook.h free of <cuda_runtime.h>.
    setCudaStreamProvider([](int32_t deviceIndex) -> void* {
      void* stream = nullptr;
      TORCH_ERROR_CODE_CHECK(
          aoti_torch_get_current_cuda_stream(deviceIndex, &stream));
      return stream;
    });
#endif
  }
};

const TorchHookRegistrar gTorchHookRegistrar;

} // namespace
} // namespace facebook::torchcodec
