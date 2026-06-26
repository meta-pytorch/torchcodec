// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// Zero-copy conversions between the torch-free core tensor type (tc::Tensor)
// and torch::stable::Tensor.
//
// The core is being migrated to tc::Tensor. The PyTorch-facing boundaries (the
// custom-ops adapter, and CUDA device interfaces that still use torch
// internally during migration) include this header to convert at the seam.
// This header is ONLY compiled in translation units that already build against
// torch with TORCH_TARGET_VERSION defined (custom_ops.cpp, the CUDA .cpp
// files); it must never be included by a torch-free translation unit.
//
// Both directions are zero-copy: the produced tensor shares the source's
// storage, and a deleter holds a reference to the source so the buffer outlives
// every view of it.

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <vector>

#include "StableABICompat.h"
#include "TCTensor.h"

namespace facebook::torchcodec {

inline torch::headeronly::ScalarType toStableDtype(tc::ScalarType dtype) {
  switch (dtype) {
    case tc::ScalarType::UInt8:
      return kStableUInt8;
    case tc::ScalarType::UInt16:
      return kStableUInt16;
    case tc::ScalarType::Int32:
      return kStableInt32;
    case tc::ScalarType::Int64:
      return kStableInt64;
    case tc::ScalarType::Float32:
      return kStableFloat32;
    case tc::ScalarType::Float64:
      return kStableFloat64;
    case tc::ScalarType::Bool:
      return kStableBool;
  }
  STD_TORCH_CHECK(false, "toStableDtype: unknown dtype");
}

inline tc::ScalarType fromStableDtype(torch::headeronly::ScalarType dtype) {
  if (dtype == kStableUInt8) {
    return tc::ScalarType::UInt8;
  }
  if (dtype == kStableUInt16) {
    return tc::ScalarType::UInt16;
  }
  if (dtype == kStableInt32) {
    return tc::ScalarType::Int32;
  }
  if (dtype == kStableInt64) {
    return tc::ScalarType::Int64;
  }
  if (dtype == kStableFloat32) {
    return tc::ScalarType::Float32;
  }
  if (dtype == kStableFloat64) {
    return tc::ScalarType::Float64;
  }
  if (dtype == kStableBool) {
    return tc::ScalarType::Bool;
  }
  STD_TORCH_CHECK(false, "fromStableDtype: unsupported dtype");
}

inline StableDevice toStableDevice(tc::Device device) {
  StableDeviceType type =
      device.type == tc::DeviceType::CUDA ? kStableCUDA : kStableCPU;
  return StableDevice(type, static_cast<int32_t>(device.index));
}

inline tc::Device fromStableDevice(const StableDevice& device) {
  if (device.type() == kStableCUDA) {
    return tc::Device(tc::DeviceType::CUDA, static_cast<int32_t>(device.index()));
  }
  return tc::Device(tc::DeviceType::CPU, static_cast<int32_t>(device.index()));
}

// tc::Tensor -> torch::stable::Tensor (zero-copy; keeps the tc storage alive).
inline torch::stable::Tensor toStable(const tc::Tensor& t) {
  if (!t.defined()) {
    return torch::stable::Tensor();
  }
  // Hold a reference to the source storage until torch frees the blob.
  tc::Tensor keepAlive = t;
  return torch::stable::from_blob(
      t.mutable_data_ptr(),
      t.sizes(),
      t.strides(),
      toStableDevice(t.device()),
      toStableDtype(t.scalar_type()),
      [keepAlive](void*) mutable { /* releases keepAlive on storage free */ });
}

// torch::stable::Tensor -> tc::Tensor (zero-copy; keeps the torch tensor alive).
inline tc::Tensor fromStable(const torch::stable::Tensor& t) {
  std::vector<int64_t> sizes(t.sizes().begin(), t.sizes().end());
  std::vector<int64_t> strides(t.strides().begin(), t.strides().end());
  torch::stable::Tensor keepAlive = t;
  return tc::from_blob(
      t.mutable_data_ptr(),
      std::move(sizes),
      std::move(strides),
      fromStableDtype(t.scalar_type()),
      fromStableDevice(t.device()),
      [keepAlive](void*) mutable { /* releases keepAlive on storage free */ });
}

} // namespace facebook::torchcodec
