// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Symbol visibility for the shared library
#ifdef _WIN32
#define TORCHCODEC_API __declspec(dllexport)
#else
#define TORCHCODEC_API __attribute__((visibility("default")))
#endif

// Flag meant to be used for any API that third-party libraries may call.
// It ensures the API symbol is always public.
#ifdef _WIN32
#define TORCHCODEC_THIRD_PARTY_API
#else
#define TORCHCODEC_THIRD_PARTY_API __attribute__((visibility("default")))
#endif

// Index error check - throws std::out_of_range which pybind11 maps to
// IndexError Use this for index validation errors that should raise IndexError
// in Python
#define STABLE_CHECK_INDEX(cond, msg)            \
  do {                                           \
    if (!(cond)) {                               \
      throw std::out_of_range(std::string(msg)); \
    }                                            \
  } while (false)

namespace facebook::torchcodec {

// ============================================================================
// Tensor types
// ============================================================================
using StableTensor = torch::stable::Tensor;
using StableScalarType = torch::headeronly::ScalarType;
using StableIntArrayRef = torch::headeronly::IntHeaderOnlyArrayRef;
using StableLayout = torch::headeronly::Layout;
using StableMemoryFormat = torch::headeronly::MemoryFormat;
using StableDeviceIndex = torch::stable::accelerator::DeviceIndex;

// ============================================================================
// Device types
// ============================================================================
using StableDevice = torch::stable::Device;
using StableDeviceType = torch::headeronly::DeviceType;

// DeviceGuard for CUDA context management
using StableDeviceGuard = torch::stable::accelerator::DeviceGuard;

// Device type constants
constexpr auto kStableCPU = torch::headeronly::DeviceType::CPU;
constexpr auto kStableCUDA = torch::headeronly::DeviceType::CUDA;

// ============================================================================
// Scalar type constants
// ============================================================================
constexpr auto kStableUInt8 = torch::headeronly::ScalarType::Byte;
constexpr auto kStableInt8 = torch::headeronly::ScalarType::Char;
constexpr auto kStableInt16 = torch::headeronly::ScalarType::Short;
constexpr auto kStableInt32 = torch::headeronly::ScalarType::Int;
constexpr auto kStableInt64 = torch::headeronly::ScalarType::Long;
constexpr auto kStableFloat16 = torch::headeronly::ScalarType::Half;
constexpr auto kStableFloat32 = torch::headeronly::ScalarType::Float;
constexpr auto kStableFloat64 = torch::headeronly::ScalarType::Double;
constexpr auto kStableBool = torch::headeronly::ScalarType::Bool;

// Layout constants
constexpr auto kStableStrided = torch::headeronly::Layout::Strided;

// ============================================================================
// Helper functions wrapping torch::stable ops
// ============================================================================

// aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
inline StableTensor stablePermute(
    const StableTensor& self,
    std::vector<int64_t> dims) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(dims)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::permute", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<StableTensor>(stack[0]);
}

// aten::cat(Tensor[] tensors, int dim=0) -> Tensor
inline StableTensor stableCat(
    const std::vector<StableTensor>& tensors,
    int64_t dim) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(tensors), torch::stable::detail::from(dim)};
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::cat", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<StableTensor>(stack[0]);
}

// aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor
inline StableTensor stableRot90(
    const StableTensor& self,
    int k,
    int64_t dim0,
    int64_t dim1) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(static_cast<int64_t>(k)),
      torch::stable::detail::from(std::vector<int64_t>{dim0, dim1})};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::rot90", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<StableTensor>(stack[0]);
}

inline const char* deviceTypeName(StableDeviceType deviceType) {
  switch (deviceType) {
    case kStableCPU:
      return "cpu";
    case kStableCUDA:
      return "cuda";
    default:
      return "unknown";
  }
}

inline std::string intArrayRefToString(StableIntArrayRef arr) {
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (i > 0)
      ss << ", ";
    ss << arr[i];
  }
  ss << "]";
  return ss.str();
}

} // namespace facebook::torchcodec
