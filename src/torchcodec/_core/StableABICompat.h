// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// PyTorch Stable ABI Compatibility Header
// ===========================================================================
//
// This header provides compatibility types and macros for using PyTorch's
// stable ABI API. It is designed to replace the standard PyTorch C++ APIs
// (torch::, at::, c10::) with their stable ABI equivalents.
//
// Target PyTorch version: 2.11+
//
// Note: TORCH_TARGET_VERSION is set to 0x020b000000000000 (PyTorch 2.11) in
// CMakeLists.txt. This ensures we only use stable ABI features available in
// PyTorch 2.11+, providing forward compatibility when building against newer
// PyTorch versions.

// Include stable ABI headers
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>

#include <array>
#include <vector>

// ===========================================================================
// Error Handling Macro
// ===========================================================================
// Replacement for TORCH_CHECK() that works with stable ABI.
// Uses STD_TORCH_CHECK from the stable ABI headers.
// Note: Unlike TORCH_CHECK, this always requires a message argument.

#define STABLE_CHECK(cond, ...) STD_TORCH_CHECK(cond, __VA_ARGS__)

// Index error check - throws std::out_of_range which pybind11 maps to IndexError
// Use this for index validation errors that should raise IndexError in Python
#define STABLE_CHECK_INDEX(cond, msg)              \
  do {                                             \
    if (!(cond)) {                                 \
      throw std::out_of_range(std::string(msg));   \
    }                                              \
  } while (false)

// ===========================================================================
// Symbol Visibility Macro
// ===========================================================================
// Cross-platform macro for symbol visibility (replaces TORCH_API).
// On Windows, uses __declspec(dllexport).
// On GCC/Clang, uses __attribute__((visibility("default"))).

#ifdef _WIN32
#define TORCHCODEC_API __declspec(dllexport)
#else
#define TORCHCODEC_API __attribute__((visibility("default")))
#endif

// ===========================================================================
// Type Aliases
// ===========================================================================
// Convenient aliases for stable ABI types

namespace facebook::torchcodec {

// Tensor types
using StableTensor = torch::stable::Tensor;

// Device types
using StableDevice = torch::stable::Device;
using StableDeviceType = torch::headeronly::DeviceType;
using StableDeviceIndex = torch::stable::accelerator::DeviceIndex;

// Scalar types (dtype)
using StableScalarType = torch::headeronly::ScalarType;

// DeviceGuard for CUDA context management
using StableDeviceGuard = torch::stable::accelerator::DeviceGuard;

// Array reference type for sizes/strides
using StableIntArrayRef = torch::headeronly::IntHeaderOnlyArrayRef;

// Layout and MemoryFormat
using StableLayout = torch::headeronly::Layout;
using StableMemoryFormat = torch::headeronly::MemoryFormat;

// ===========================================================================
// Constants
// ===========================================================================

// Device type constants
constexpr auto kStableCPU = torch::headeronly::DeviceType::CPU;
constexpr auto kStableCUDA = torch::headeronly::DeviceType::CUDA;

// Scalar type constants (equivalents of torch::kUInt8, torch::kFloat32, etc.)
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

// ===========================================================================
// Helper Functions
// ===========================================================================

// Stable version of torch::empty()
inline StableTensor stableEmpty(
    std::initializer_list<int64_t> sizes,
    StableScalarType dtype,
    StableDevice device) {
  std::vector<int64_t> sizesVec(sizes);
  return torch::stable::empty(
      StableIntArrayRef(sizesVec.data(), sizesVec.size()),
      dtype,
      kStableStrided,
      device);
}

// Overload taking a vector
inline StableTensor stableEmpty(
    const std::vector<int64_t>& sizes,
    StableScalarType dtype,
    StableDevice device) {
  return torch::stable::empty(
      StableIntArrayRef(sizes.data(), sizes.size()),
      dtype,
      kStableStrided,
      device);
}

// Helper to create CPU tensors (for pts/duration which are always on CPU)
inline StableTensor stableEmptyCPU(
    std::initializer_list<int64_t> sizes,
    StableScalarType dtype) {
  return stableEmpty(sizes, dtype, StableDevice(kStableCPU));
}

// Stable version of tensor.copy_(src)
inline void stableCopy_(StableTensor& dst, const StableTensor& src) {
  torch::stable::copy_(dst, src);
}

// Stable version of tensor.to(device)
inline StableTensor stableTo(
    const StableTensor& tensor,
    const StableDevice& device) {
  return torch::stable::to(tensor, device);
}

// Stable version of tensor.narrow(dim, start, length)
// Note: torch::stable::narrow() requires non-const tensor reference
inline StableTensor
stableNarrow(StableTensor tensor, int64_t dim, int64_t start, int64_t length) {
  return torch::stable::narrow(tensor, dim, start, length);
}

// Stable version of tensor.contiguous()
inline StableTensor stableContiguous(const StableTensor& tensor) {
  return torch::stable::contiguous(tensor);
}

// Stable version of tensor.permute()
// Uses the dispatcher to call aten::permute
inline StableTensor stablePermute(
    const StableTensor& tensor,
    std::initializer_list<int64_t> dims) {
  std::vector<int64_t> dimsVec(dims);
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(tensor),
      torch::stable::detail::from(
          StableIntArrayRef(dimsVec.data(), dimsVec.size()))};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::permute", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<StableTensor>(stack[0]);
}

// Helper to create a scalar (0-dimensional) tensor from a double value
// Replaces torch::tensor(value, torch::dtype(torch::kFloat64))
inline StableTensor stableScalarTensor(double value) {
  StableTensor t = stableEmpty({}, kStableFloat64, StableDevice(kStableCPU));
  *t.mutable_data_ptr<double>() = value;
  return t;
}

// Helper for torch::cat() - concatenates tensors along a dimension
// Uses dispatcher to call aten::cat
inline StableTensor stableCat(
    const std::vector<StableTensor>& tensors,
    int64_t dim = 0) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(tensors), torch::stable::detail::from(dim)};
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::cat", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<StableTensor>(stack[0]);
}

// Helper for tensor.is_contiguous()
// Uses the stable ABI tensor method directly
inline bool stableIsContiguous(const StableTensor& tensor) {
  return tensor.is_contiguous();
}

// Helper for tensor[index] - selects along a dimension
// Uses torch::stable::select from PyTorch's stable ops
inline StableTensor
stableSelect(const StableTensor& tensor, int64_t dim, int64_t index) {
  return torch::stable::select(tensor, dim, index);
}

// Helper to get a human-readable name for a scalar type
inline const char* scalarTypeName(StableScalarType dtype) {
  switch (dtype) {
    case torch::headeronly::ScalarType::Byte:
      return "uint8";
    case torch::headeronly::ScalarType::Char:
      return "int8";
    case torch::headeronly::ScalarType::Short:
      return "int16";
    case torch::headeronly::ScalarType::Int:
      return "int32";
    case torch::headeronly::ScalarType::Long:
      return "int64";
    case torch::headeronly::ScalarType::Half:
      return "float16";
    case torch::headeronly::ScalarType::Float:
      return "float32";
    case torch::headeronly::ScalarType::Double:
      return "float64";
    case torch::headeronly::ScalarType::Bool:
      return "bool";
    default:
      return "unknown";
  }
}

// Helper to get a human-readable name for a device type
inline const char* deviceTypeName(StableDeviceType dtype) {
  switch (dtype) {
    case torch::headeronly::DeviceType::CPU:
      return "cpu";
    case torch::headeronly::DeviceType::CUDA:
      return "cuda";
    default:
      return "unknown";
  }
}

// Helper to convert IntArrayRef to a string for error messages
inline std::string intArrayRefToString(const StableIntArrayRef& arr) {
  std::string result = "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (i > 0) {
      result += ", ";
    }
    result += std::to_string(arr[i]);
  }
  result += "]";
  return result;
}

} // namespace facebook::torchcodec