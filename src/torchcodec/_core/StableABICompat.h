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

// Check macro similar to TORCH_CHECK but using STD_TORCH_CHECK style
#define STABLE_CHECK(cond, ...)                                    \
  do {                                                             \
    if (!(cond)) {                                                 \
      std::ostringstream __stable_check_ss;                        \
      __stable_check_ss << "STABLE_CHECK failed: " << __VA_ARGS__; \
      throw std::runtime_error(__stable_check_ss.str());           \
    }                                                              \
  } while (false)

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

// Permute tensor dimensions. Since permute is not in the stable API,
// we implement it via successive transpose operations.
inline StableTensor stablePermute(
    const StableTensor& self,
    std::initializer_list<int64_t> dims) {
  std::vector<int64_t> perm(dims);
  int64_t ndim = self.dim();

  // Build the permutation using transpositions.
  // We apply a sequence of transpose operations to achieve the desired
  // permutation.
  StableTensor result = torch::stable::clone(self);

  // Track current position of each original dimension
  std::vector<int64_t> pos(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    pos[i] = i;
  }

  // For each target position, find where the desired dimension currently is
  // and transpose it into place.
  for (int64_t i = 0; i < ndim; ++i) {
    // Find where perm[i] currently is
    int64_t j = i;
    for (int64_t k = i; k < ndim; ++k) {
      if (pos[k] == perm[i]) {
        j = k;
        break;
      }
    }
    if (i != j) {
      result = torch::stable::transpose(result, i, j);
      std::swap(pos[i], pos[j]);
    }
  }
  return result;
}

// Concatenate tensors along a dimension.
// Since cat is not in the stable API, we implement it manually:
// allocate output, then copy_ slices via narrow.
inline StableTensor stableCat(
    const std::vector<StableTensor>& tensors,
    int64_t dim) {
  STD_TORCH_CHECK(!tensors.empty(), "stableCat: tensor list must not be empty");

  // Compute the total size along the cat dimension
  int64_t totalSize = 0;
  for (const auto& t : tensors) {
    totalSize += t.sizes()[dim];
  }

  // Build the output shape
  auto firstSizes = tensors[0].sizes();
  std::vector<int64_t> outShape(firstSizes.begin(), firstSizes.end());
  outShape[dim] = totalSize;

  StableTensor result = torch::stable::empty(
      outShape, tensors[0].scalar_type(), kStableStrided, tensors[0].device());

  // Copy each tensor into the right slice
  int64_t offset = 0;
  for (const auto& t : tensors) {
    int64_t len = t.sizes()[dim];
    auto slice = torch::stable::narrow(result, dim, offset, len);
    torch::stable::copy_(slice, t, /*non_blocking=*/std::nullopt);
    offset += len;
  }

  return result;
}

// ============================================================================
// String helpers for error messages
// ============================================================================

inline const char* scalarTypeName(StableScalarType dtype) {
  switch (dtype) {
    case kStableUInt8:
      return "uint8";
    case kStableInt8:
      return "int8";
    case kStableInt16:
      return "int16";
    case kStableInt32:
      return "int32";
    case kStableInt64:
      return "int64";
    case kStableFloat16:
      return "float16";
    case kStableFloat32:
      return "float32";
    case kStableFloat64:
      return "float64";
    case kStableBool:
      return "bool";
    default:
      return "unknown";
  }
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
