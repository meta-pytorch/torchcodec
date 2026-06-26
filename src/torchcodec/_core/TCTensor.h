// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// tc::Tensor is torchcodec's lightweight, PyTorch-free tensor type. It owns a
// reference-counted buffer plus shape/strides/dtype/device metadata and
// implements the small op surface the core needs (the exact set previously
// satisfied by torch::stable). It exports to DLPack so torch / numpy / cupy /
// jax can all consume decoded frames zero-copy.
//
// Scope note: CPU is fully implemented. CUDA allocation / device transfer /
// elementwise kernels are stubbed to throw and are wired up in the CUDA part of
// Phase A.

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dlpack.h"

namespace facebook::torchcodec::tc {

// Scalar dtypes the core uses. Values are arbitrary; mapping to DLPack and to
// torch scalar types happens at the edges.
enum class ScalarType {
  UInt8,
  UInt16,
  Int32,
  Int64,
  Float32,
  Float64,
  Bool,
};

enum class DeviceType {
  CPU,
  CUDA,
  XPU,
};

// Mirrors the small surface of torch::stable::Device (type()/index()) so core
// call sites that were written against it need minimal churn.
class Device {
 public:
  Device() = default;
  /* implicit */ Device(DeviceType type, int32_t index = 0)
      : type_(type), index_(index) {}

  DeviceType type() const {
    return type_;
  }
  int32_t index() const {
    return index_;
  }

  bool operator==(const Device& other) const {
    return type_ == other.type_ && index_ == other.index_;
  }
  bool operator!=(const Device& other) const {
    return !(*this == other);
  }

 private:
  DeviceType type_ = DeviceType::CPU;
  int32_t index_ = 0;
};

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kXPU = DeviceType::XPU;

// Size in bytes of one element of the given dtype.
int64_t elementSize(ScalarType dtype);

// Deleter signature compatible with torch::stable::from_blob.
using DeleterFn = std::function<void(void*)>;

// ---- Pluggable storage allocator ----
//
// tc::Tensor never calls cudaMalloc itself, and does not reimplement a caching
// allocator. Instead, allocation of the raw storage block is delegated to a
// process-wide hook. This is how we honor "use torch's CUDA caching allocator
// when torch is present":
//   - torch present  -> the torch adapter registers an allocator that allocates
//                       via torch::stable::empty (torch's caching allocator on
//                       CUDA) and returns storage that owns the torch tensor.
//   - torch absent + CUDA -> a simple cudaMalloc/cudaFree allocator (registered
//                       by the torch-free GPU build).
//   - torch absent + CPU  -> the built-in malloc fallback (no hook needed).
//
// The hook returns the owning storage for `numBytes` bytes on `device`. It is
// expected to be installed once at module init (not thread-safe to swap).
using AllocFn = std::function<
    std::shared_ptr<void>(int64_t numBytes, ScalarType dtype, Device device)>;

// Install (or, with nullptr, clear) the global storage allocator.
void setAllocator(AllocFn fn);

// True if a custom allocator has been installed.
bool hasAllocator();

class Tensor; // forward decl for the backend hook below

// ---- Pluggable compute backend (for non-CPU devices) ----
//
// tc::Tensor implements all ops natively for CPU. For non-CPU devices it does
// NOT reimplement GPU kernels; instead each data-touching op is dispatched to a
// per-device-type backend registered at module init. This mirrors the allocator
// hook and keeps the torch-free GPU path pluggable:
//   - torch present -> the adapter registers a CUDA backend that runs each op
//                      via torch (toStable -> torch op -> fromStable).
//   - torch-free GPU (later) -> register a backend backed by CUDA kernels.
//
// Metadata-only views (narrow/select/permute) never need a backend; they only
// manipulate shape/strides and work on any device.
struct DeviceBackend {
  std::function<void(Tensor& dst, const Tensor& src)> copy_;
  std::function<Tensor(const Tensor& self, ScalarType dtype)> toDtype;
  std::function<Tensor(const Tensor& self, Device device)> toDevice;
  std::function<Tensor(const Tensor& self, double other)> div;
  std::function<Tensor(const Tensor& self)> contiguous;
  std::function<Tensor(const std::vector<Tensor>& tensors, int64_t dim)> cat;
  std::function<Tensor(const Tensor& self, int64_t k, int64_t d0, int64_t d1)>
      rot90;
};

// Register (or, with a default-constructed backend, leaving fields null,
// effectively clear) the compute backend for a device type.
void registerDeviceBackend(DeviceType deviceType, DeviceBackend backend);

// Returns the registered backend for a device type, or nullptr if none.
const DeviceBackend* getDeviceBackend(DeviceType deviceType);

class Tensor {
 public:
  // Default-constructed tensor is "undefined" (no storage). Mirrors
  // torch::stable::Tensor's default state; defined() reports validity.
  Tensor() = default;

  // Low-level constructor adopting an existing storage block.
  Tensor(
      std::shared_ptr<void> storage,
      void* dataBase,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides,
      ScalarType dtype,
      Device device,
      int64_t storageOffsetElems = 0);

  bool defined() const {
    return static_cast<bool>(storage_) || dataBase_ != nullptr;
  }

  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }
  const std::vector<int64_t>& strides() const {
    return strides_;
  }
  int64_t size(int64_t dim) const {
    return sizes_.at(normalizeDim(dim));
  }
  int64_t stride(int64_t dim) const {
    return strides_.at(normalizeDim(dim));
  }
  int64_t dim() const {
    return static_cast<int64_t>(sizes_.size());
  }
  int64_t numel() const;
  ScalarType scalar_type() const {
    return dtype_;
  }
  Device device() const {
    return device_;
  }
  int64_t element_size() const {
    return elementSize(dtype_);
  }
  bool is_contiguous() const;

  // Byte pointer to element [0...]. Honors storage offset.
  void* mutable_data_ptr() const {
    return static_cast<char*>(dataBase_) + storageOffsetElems_ * element_size();
  }
  template <typename T>
  T* mutable_data_ptr() const {
    return reinterpret_cast<T*>(mutable_data_ptr());
  }
  template <typename T>
  const T* const_data_ptr() const {
    return reinterpret_cast<const T*>(mutable_data_ptr());
  }

  // The underlying storage (kept so views and DLPack export can extend
  // lifetime).
  const std::shared_ptr<void>& storage() const {
    return storage_;
  }
  int64_t storage_offset() const {
    return storageOffsetElems_;
  }

 private:
  int64_t normalizeDim(int64_t dim) const {
    int64_t n = static_cast<int64_t>(sizes_.size());
    return dim < 0 ? dim + n : dim;
  }

  std::shared_ptr<void> storage_;
  void* dataBase_ = nullptr; // base of storage (offset applied via accessors)
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_; // in elements
  ScalarType dtype_ = ScalarType::Float32;
  Device device_;
  int64_t storageOffsetElems_ = 0;
};

// ---- Factory / op surface (mirrors the torch::stable subset the core uses) ----

Tensor empty(
    std::vector<int64_t> sizes,
    ScalarType dtype,
    Device device = Device{});

// 0-dim or N-dim tensor filled with a single value.
Tensor full(std::vector<int64_t> sizes, double value, ScalarType dtype);

// Adopt an external buffer, calling `deleter(data)` when the last reference
// dies. Strides are in elements.
Tensor from_blob(
    void* data,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides,
    ScalarType dtype,
    Device device,
    DeleterFn deleter);

// In-place ops.
void copy_(Tensor& dst, const Tensor& src); // casts if dtypes differ
void zero_(Tensor& t);

// dtype / device casts (return new tensors).
Tensor to(const Tensor& self, ScalarType dtype);
Tensor to(const Tensor& self, Device device);

// Views (share storage where possible).
Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length);
Tensor select(const Tensor& self, int64_t dim, int64_t index);
Tensor permute(const Tensor& self, std::vector<int64_t> dims);

// Materializing ops.
Tensor contiguous(const Tensor& self);
Tensor cat(const std::vector<Tensor>& tensors, int64_t dim);
Tensor rot90(const Tensor& self, int64_t k, int64_t dim0, int64_t dim1);
Tensor div(const Tensor& self, double other);

// Shorthand for select(t, 0, index), i.e. t[index].
inline Tensor selectRow(const Tensor& t, int64_t index) {
  return select(t, 0, index);
}

// ---- DLPack interop ----

// Produce an owning DLManagedTensor that shares this tensor's storage. The
// caller (typically a PyCapsule named "dltensor") owns it and must invoke its
// deleter exactly once.
DLManagedTensor* toDLPack(const Tensor& t);

// Adopt a DLManagedTensor (steals ownership; its deleter is called when the
// resulting tensor's storage is released).
Tensor fromDLPack(DLManagedTensor* managed);

} // namespace facebook::torchcodec::tc
