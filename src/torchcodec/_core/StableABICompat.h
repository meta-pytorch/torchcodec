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

#include <torch/headeronly/core/TensorAccessor.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Symbol visibility for the shared library
#ifdef _WIN32
#define FORCE_PUBLIC_VISIBILITY __declspec(dllexport)
#else
#define FORCE_PUBLIC_VISIBILITY __attribute__((visibility("default")))
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

// Device types
using StableDevice = torch::stable::Device;
using StableDeviceType = torch::headeronly::DeviceType;

// DeviceGuard for CUDA context management
using StableDeviceGuard = torch::stable::accelerator::DeviceGuard;

// Device type constants
constexpr auto kStableCPU = torch::headeronly::DeviceType::CPU;
constexpr auto kStableCUDA = torch::headeronly::DeviceType::CUDA;
constexpr auto kStableXPU = torch::headeronly::DeviceType::XPU;

// Scalar type constants
constexpr auto kStableUInt8 = torch::headeronly::ScalarType::Byte;
constexpr auto kStableInt64 = torch::headeronly::ScalarType::Long;
constexpr auto kStableFloat32 = torch::headeronly::ScalarType::Float;
constexpr auto kStableFloat64 = torch::headeronly::ScalarType::Double;
constexpr auto kStableBool = torch::headeronly::ScalarType::Bool;

// Note: the magic use of torch_call_dispatcher is what is officially
// recommended
// https://github.com/pytorch/pytorch/blob/89f3759429b96a8693b698f013990240bb4e25b3/docs/source/notes/libtorch_stable_abi.md?plain=1#L221
// It allows us to make an ABI-stable call to an op that isn't officially in the
// stable ABI. Some of these ops currently include permute, rot90, etc.
// See also how xformers relies on it:
// https://github.com/facebookresearch/xformers/blob/720adff2b021f6f43957718514f5be3d10e36fb1/xformers/csrc/pt_stable_utils.h#L85

// TODO_STABLE_ABI: upstream?
inline torch::stable::Tensor stablePermute(
    const torch::stable::Tensor& self,
    std::vector<int64_t> dims) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(dims)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::permute", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// TODO_STABLE_ABI: upstream?
inline torch::stable::Tensor stableCat(
    const std::vector<torch::stable::Tensor>& tensors,
    int64_t dim) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(tensors), torch::stable::detail::from(dim)};
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::cat", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// TODO_STABLE_ABI: upstream?
inline torch::stable::Tensor stableRot90(
    const torch::stable::Tensor& self,
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
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// Shorthand for torch::stable::select(tensor, 0, index), i.e. tensor[index].
inline torch::stable::Tensor selectRow(
    const torch::stable::Tensor& tensor,
    int64_t index) {
  return torch::stable::select(tensor, 0, index);
}

template <typename T, size_t N>
torch::headeronly::HeaderOnlyTensorAccessor<T, N> mutableAccessor(
    torch::stable::Tensor& tensor) {
  return torch::headeronly::HeaderOnlyTensorAccessor<T, N>(
      tensor.mutable_data_ptr<T>(),
      tensor.sizes().data(),
      tensor.strides().data());
}

template <typename T, size_t N>
torch::headeronly::HeaderOnlyTensorAccessor<const T, N> constAccessor(
    const torch::stable::Tensor& tensor) {
  return torch::headeronly::HeaderOnlyTensorAccessor<const T, N>(
      tensor.const_data_ptr<T>(),
      tensor.sizes().data(),
      tensor.strides().data());
}

// Copy row srcIndex from srcTensor into row dstIndex of dstTensor.
inline void copyFrame(
    torch::stable::Tensor& dstTensor,
    int64_t dstIndex,
    const torch::stable::Tensor& srcTensor,
    int64_t srcIndex) {
  auto dst = selectRow(dstTensor, dstIndex);
  torch::stable::copy_(dst, selectRow(srcTensor, srcIndex));
}

// TODO_STABLE_ABI: this should probably be natively supported by torch::stable.
// Consider upstreaming.
inline const char* deviceTypeName(StableDeviceType deviceType) {
  switch (deviceType) {
    case kStableCPU:
      return "cpu";
    case kStableCUDA:
      return "cuda";
    case kStableXPU:
      return "xpu";
    default:
      return "unknown";
  }
}

// TODO_STABLE_ABI: This is needed to properly print shape info in error
// messages. There should probably be a better native way to support it, e.g.
// torch::headeronly::IntHeaderOnlyArrayRef probably needs to support the `<<`
// operator. Consider upstreaming.
inline std::string intArrayRefToString(
    torch::headeronly::IntHeaderOnlyArrayRef arr) {
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
