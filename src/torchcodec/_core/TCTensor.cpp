// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "TCTensor.h"

#include <cstring>
#include <map>
#include <numeric>

namespace facebook::torchcodec::tc {

namespace {

[[noreturn]] void fail(const std::string& msg) {
  throw std::runtime_error("tc::Tensor: " + msg);
}

void checkCpu(const Device& device, const char* what) {
  if (device.type() != DeviceType::CPU) {
    fail(
        std::string(what) +
        " is only implemented for CPU tensors (CUDA support is the CUDA part "
        "of Phase A).");
  }
}

std::vector<int64_t> contiguousStrides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (int64_t d = static_cast<int64_t>(sizes.size()) - 1; d >= 0; --d) {
    strides[d] = acc;
    acc *= sizes[d];
  }
  return strides;
}

int64_t numelOf(const std::vector<int64_t>& sizes) {
  int64_t n = 1;
  for (auto s : sizes) {
    n *= s;
  }
  return n;
}

// Process-wide storage allocator hook (see TCTensor.h). Default is empty, in
// which case CPU allocation uses malloc and non-CPU allocation throws.
AllocFn g_allocator;

// Process-wide compute backends, keyed by device type (see TCTensor.h).
std::map<DeviceType, DeviceBackend>& backendRegistry() {
  static std::map<DeviceType, DeviceBackend> registry;
  return registry;
}

// Fetch the backend for a non-CPU device, failing clearly if none registered.
const DeviceBackend& requireBackend(Device device, const char* op) {
  const DeviceBackend* backend = getDeviceBackend(device.type());
  if (backend == nullptr) {
    fail(
        std::string(op) +
        ": no compute backend registered for this non-CPU device (GPU ops "
        "require torch, or a registered tc backend)");
  }
  return *backend;
}

// Allocate an uninitialized CPU storage block via malloc.
std::shared_ptr<void> allocCpuStorage(int64_t numBytes) {
  if (numBytes == 0) {
    // Non-null sentinel so defined() is true for empty tensors.
    return std::shared_ptr<void>(reinterpret_cast<void*>(1), [](void*) {});
  }
  void* p = ::operator new(static_cast<size_t>(numBytes));
  return std::shared_ptr<void>(p, [](void* q) { ::operator delete(q); });
}

// Allocate storage for a tensor, routing through the installed allocator hook
// when present (e.g. torch's caching allocator), else falling back to malloc
// for CPU. Non-CPU allocation without a hook is an error.
std::shared_ptr<void>
allocStorage(int64_t numBytes, ScalarType dtype, Device device) {
  if (g_allocator) {
    return g_allocator(numBytes, dtype, device);
  }
  checkCpu(
      device,
      "empty (no allocator installed; non-CPU allocation needs an allocator "
      "hook, e.g. torch's)");
  return allocCpuStorage(numBytes);
}

// Read element at byte pointer as a double (exact for all supported dtypes
// except Int64 with magnitudes > 2^53, which the casting paths don't exercise).
double loadAsDouble(const void* p, ScalarType dt) {
  switch (dt) {
    case ScalarType::UInt8:
      return static_cast<double>(*reinterpret_cast<const uint8_t*>(p));
    case ScalarType::UInt16:
      return static_cast<double>(*reinterpret_cast<const uint16_t*>(p));
    case ScalarType::Int32:
      return static_cast<double>(*reinterpret_cast<const int32_t*>(p));
    case ScalarType::Int64:
      return static_cast<double>(*reinterpret_cast<const int64_t*>(p));
    case ScalarType::Float32:
      return static_cast<double>(*reinterpret_cast<const float*>(p));
    case ScalarType::Float64:
      return *reinterpret_cast<const double*>(p);
    case ScalarType::Bool:
      return *reinterpret_cast<const uint8_t*>(p) ? 1.0 : 0.0;
  }
  fail("unknown dtype in loadAsDouble");
}

void storeFromDouble(void* p, ScalarType dt, double v) {
  switch (dt) {
    case ScalarType::UInt8:
      *reinterpret_cast<uint8_t*>(p) = static_cast<uint8_t>(v);
      return;
    case ScalarType::UInt16:
      *reinterpret_cast<uint16_t*>(p) = static_cast<uint16_t>(v);
      return;
    case ScalarType::Int32:
      *reinterpret_cast<int32_t*>(p) = static_cast<int32_t>(v);
      return;
    case ScalarType::Int64:
      *reinterpret_cast<int64_t*>(p) = static_cast<int64_t>(v);
      return;
    case ScalarType::Float32:
      *reinterpret_cast<float*>(p) = static_cast<float>(v);
      return;
    case ScalarType::Float64:
      *reinterpret_cast<double*>(p) = v;
      return;
    case ScalarType::Bool:
      *reinterpret_cast<uint8_t*>(p) = v != 0.0 ? 1 : 0;
      return;
  }
  fail("unknown dtype in storeFromDouble");
}

// Walk every multi-index of `sizes` in row-major order, invoking fn(idx).
template <typename Fn>
void forEachIndex(const std::vector<int64_t>& sizes, Fn&& fn) {
  int64_t n = numelOf(sizes);
  if (n == 0) {
    return;
  }
  std::vector<int64_t> idx(sizes.size(), 0);
  for (int64_t lin = 0; lin < n; ++lin) {
    fn(idx);
    for (int64_t d = static_cast<int64_t>(sizes.size()) - 1; d >= 0; --d) {
      if (++idx[d] < sizes[d]) {
        break;
      }
      idx[d] = 0;
    }
  }
}

// Byte pointer to element `idx` of tensor `t`.
char* elemPtr(const Tensor& t, const std::vector<int64_t>& idx) {
  int64_t off = 0;
  const auto& strides = t.strides();
  for (size_t d = 0; d < idx.size(); ++d) {
    off += idx[d] * strides[d];
  }
  return static_cast<char*>(t.mutable_data_ptr()) + off * t.element_size();
}

} // namespace

int64_t elementSize(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::UInt8:
    case ScalarType::Bool:
      return 1;
    case ScalarType::UInt16:
      return 2;
    case ScalarType::Int32:
    case ScalarType::Float32:
      return 4;
    case ScalarType::Int64:
    case ScalarType::Float64:
      return 8;
  }
  fail("unknown dtype in elementSize");
}

Tensor::Tensor(
    std::shared_ptr<void> storage,
    void* dataBase,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides,
    ScalarType dtype,
    Device device,
    int64_t storageOffsetElems)
    : storage_(std::move(storage)),
      dataBase_(dataBase),
      sizes_(std::move(sizes)),
      strides_(std::move(strides)),
      dtype_(dtype),
      device_(device),
      storageOffsetElems_(storageOffsetElems) {}

Device::Device(const std::string& deviceStr) {
  std::string typeStr = deviceStr;
  int32_t index = 0;
  auto colon = deviceStr.find(':');
  if (colon != std::string::npos) {
    typeStr = deviceStr.substr(0, colon);
    index = std::stoi(deviceStr.substr(colon + 1));
  }
  if (typeStr == "cpu") {
    type_ = DeviceType::CPU;
  } else if (typeStr == "cuda") {
    type_ = DeviceType::CUDA;
  } else if (typeStr == "xpu") {
    type_ = DeviceType::XPU;
  } else {
    fail("Device: unknown device string '" + deviceStr + "'");
  }
  index_ = index;
}

int64_t Tensor::numel() const {
  return numelOf(sizes_);
}

bool Tensor::is_contiguous() const {
  return strides_ == contiguousStrides(sizes_);
}

void setAllocator(AllocFn fn) {
  g_allocator = std::move(fn);
}

bool hasAllocator() {
  return static_cast<bool>(g_allocator);
}

void registerDeviceBackend(DeviceType deviceType, DeviceBackend backend) {
  backendRegistry()[deviceType] = std::move(backend);
}

const DeviceBackend* getDeviceBackend(DeviceType deviceType) {
  auto& registry = backendRegistry();
  auto it = registry.find(deviceType);
  return it == registry.end() ? nullptr : &it->second;
}

Tensor empty(std::vector<int64_t> sizes, ScalarType dtype, Device device) {
  int64_t n = numelOf(sizes);
  int64_t bytes = n * elementSize(dtype);
  auto storage = allocStorage(bytes, dtype, device);
  void* base = storage.get();
  auto strides = contiguousStrides(sizes);
  return Tensor(
      std::move(storage),
      base,
      std::move(sizes),
      std::move(strides),
      dtype,
      device);
}

Tensor full(std::vector<int64_t> sizes, double value, ScalarType dtype) {
  Tensor t = empty(sizes, dtype, Device{});
  int64_t es = t.element_size();
  forEachIndex(t.sizes(), [&](const std::vector<int64_t>& idx) {
    storeFromDouble(elemPtr(t, idx), dtype, value);
  });
  (void)es;
  return t;
}

Tensor from_blob(
    void* data,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides,
    ScalarType dtype,
    Device device,
    DeleterFn deleter) {
  // Storage holds `data`; its custom deleter invokes the caller's deleter.
  std::shared_ptr<void> storage(data, [deleter = std::move(deleter)](void* p) {
    if (deleter) {
      deleter(p);
    }
  });
  void* base = data;
  return Tensor(
      std::move(storage),
      base,
      std::move(sizes),
      std::move(strides),
      dtype,
      device);
}

void zero_(Tensor& t) {
  if (t.device().type() != DeviceType::CPU) {
    requireBackend(t.device(), "zero_").zero_(t);
    return;
  }
  if (t.is_contiguous()) {
    std::memset(t.mutable_data_ptr(), 0, t.numel() * t.element_size());
    return;
  }
  forEachIndex(t.sizes(), [&](const std::vector<int64_t>& idx) {
    std::memset(elemPtr(t, idx), 0, t.element_size());
  });
}

void copy_(Tensor& dst, const Tensor& src) {
  // If either side is non-CPU, dispatch to that device's backend (it handles
  // same-device and cross-device H2D/D2H copies).
  if (dst.device().type() != DeviceType::CPU ||
      src.device().type() != DeviceType::CPU) {
    Device computeDevice =
        dst.device().type() != DeviceType::CPU ? dst.device() : src.device();
    requireBackend(computeDevice, "copy_").copy_(dst, src);
    return;
  }
  if (dst.sizes() != src.sizes()) {
    fail("copy_ requires matching shapes");
  }
  // Fast path: same dtype + both contiguous -> single memcpy.
  if (dst.scalar_type() == src.scalar_type() && dst.is_contiguous() &&
      src.is_contiguous()) {
    std::memcpy(
        dst.mutable_data_ptr(),
        src.mutable_data_ptr(),
        dst.numel() * dst.element_size());
    return;
  }
  ScalarType sdt = src.scalar_type();
  ScalarType ddt = dst.scalar_type();
  forEachIndex(dst.sizes(), [&](const std::vector<int64_t>& idx) {
    // Need a non-const src ptr for elemPtr; src is logically read-only here.
    char* sp = elemPtr(const_cast<Tensor&>(src), idx);
    char* dp = elemPtr(dst, idx);
    storeFromDouble(dp, ddt, loadAsDouble(sp, sdt));
  });
}

Tensor to(const Tensor& self, ScalarType dtype) {
  if (self.scalar_type() == dtype) {
    return contiguous(self);
  }
  if (self.device().type() != DeviceType::CPU) {
    return requireBackend(self.device(), "to(dtype)").toDtype(self, dtype);
  }
  Tensor out = empty(self.sizes(), dtype, self.device());
  copy_(out, self);
  return out;
}

Tensor to(const Tensor& self, Device device) {
  if (self.device() == device) {
    return self;
  }
  // Generic transfer: allocate on the target device (via the allocator hook)
  // and copy_ (which dispatches H2D/D2H to the non-CPU side's backend). No
  // dedicated backend entry point needed.
  Tensor out = empty(self.sizes(), self.scalar_type(), device);
  copy_(out, self);
  return out;
}

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  int64_t n = self.dim();
  int64_t d = dim < 0 ? dim + n : dim;
  if (d < 0 || d >= n) {
    fail("narrow: dim out of range");
  }
  if (start < 0 || start + length > self.size(d)) {
    fail("narrow: range out of bounds");
  }
  std::vector<int64_t> sizes = self.sizes();
  sizes[d] = length;
  int64_t newOffset = self.storage_offset() + start * self.stride(d);
  return Tensor(
      self.storage(),
      self.mutable_data_ptr() == nullptr
          ? nullptr
          : static_cast<char*>(self.storage().get()),
      sizes,
      self.strides(),
      self.scalar_type(),
      self.device(),
      newOffset);
}

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  int64_t n = self.dim();
  int64_t d = dim < 0 ? dim + n : dim;
  if (d < 0 || d >= n) {
    fail("select: dim out of range");
  }
  int64_t i = index < 0 ? index + self.size(d) : index;
  if (i < 0 || i >= self.size(d)) {
    fail("select: index out of range");
  }
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  for (int64_t k = 0; k < n; ++k) {
    if (k == d) {
      continue;
    }
    sizes.push_back(self.size(k));
    strides.push_back(self.stride(k));
  }
  int64_t newOffset = self.storage_offset() + i * self.stride(d);
  return Tensor(
      self.storage(),
      static_cast<char*>(self.storage().get()),
      sizes,
      strides,
      self.scalar_type(),
      self.device(),
      newOffset);
}

Tensor permute(const Tensor& self, std::vector<int64_t> dims) {
  int64_t n = self.dim();
  if (static_cast<int64_t>(dims.size()) != n) {
    fail("permute: dims size mismatch");
  }
  std::vector<int64_t> sizes(n);
  std::vector<int64_t> strides(n);
  for (int64_t k = 0; k < n; ++k) {
    int64_t src = dims[k] < 0 ? dims[k] + n : dims[k];
    if (src < 0 || src >= n) {
      fail("permute: dim out of range");
    }
    sizes[k] = self.size(src);
    strides[k] = self.stride(src);
  }
  return Tensor(
      self.storage(),
      static_cast<char*>(self.storage().get()),
      sizes,
      strides,
      self.scalar_type(),
      self.device(),
      self.storage_offset());
}

Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  int64_t n = self.dim();
  std::vector<int64_t> dims(n);
  for (int64_t k = 0; k < n; ++k) {
    dims[k] = k;
  }
  int64_t a = dim0 < 0 ? dim0 + n : dim0;
  int64_t b = dim1 < 0 ? dim1 + n : dim1;
  std::swap(dims[a], dims[b]);
  return permute(self, dims);
}

Tensor contiguous(const Tensor& self) {
  if (self.is_contiguous()) {
    return self;
  }
  if (self.device().type() != DeviceType::CPU) {
    return requireBackend(self.device(), "contiguous").contiguous(self);
  }
  Tensor out = empty(self.sizes(), self.scalar_type(), self.device());
  int64_t es = self.element_size();
  forEachIndex(self.sizes(), [&](const std::vector<int64_t>& idx) {
    std::memcpy(elemPtr(out, idx), elemPtr(const_cast<Tensor&>(self), idx), es);
  });
  return out;
}

Tensor cat(const std::vector<Tensor>& tensors, int64_t dim) {
  if (tensors.empty()) {
    fail("cat: empty list");
  }
  const Tensor& first = tensors[0];
  int64_t n = first.dim();
  int64_t d = dim < 0 ? dim + n : dim;
  if (first.device().type() != DeviceType::CPU) {
    return requireBackend(first.device(), "cat").cat(tensors, dim);
  }
  std::vector<int64_t> outSizes = first.sizes();
  int64_t catSize = 0;
  for (const auto& t : tensors) {
    catSize += t.size(d);
  }
  outSizes[d] = catSize;
  Tensor out = empty(outSizes, first.scalar_type(), first.device());
  int64_t offset = 0;
  for (const auto& t : tensors) {
    Tensor dstSlice = narrow(out, d, offset, t.size(d));
    copy_(dstSlice, t);
    offset += t.size(d);
  }
  return out;
}

Tensor rot90(const Tensor& self, int64_t k, int64_t dim0, int64_t dim1) {
  // Normalize k to [0, 3].
  int64_t kk = ((k % 4) + 4) % 4;
  if (kk == 0) {
    return contiguous(self);
  }
  // rot90 = flip then transpose, repeated. Implement directly via index map on
  // a contiguous output.
  int64_t n = self.dim();
  int64_t a = dim0 < 0 ? dim0 + n : dim0;
  int64_t b = dim1 < 0 ? dim1 + n : dim1;
  if (self.device().type() != DeviceType::CPU) {
    return requireBackend(self.device(), "rot90").rot90(self, k, dim0, dim1);
  }

  Tensor cur = contiguous(self);
  for (int64_t step = 0; step < kk; ++step) {
    // One 90-degree rotation: new[i,j over a,b] = old[j, size_a-1-i].
    std::vector<int64_t> outSizes = cur.sizes();
    std::swap(outSizes[a], outSizes[b]);
    Tensor out = empty(outSizes, cur.scalar_type(), cur.device());
    int64_t es = cur.element_size();
    int64_t sizeB = cur.size(b);
    forEachIndex(out.sizes(), [&](const std::vector<int64_t>& outIdx) {
      std::vector<int64_t> inIdx = outIdx;
      // One counter-clockwise step: out[p along a, q along b] = in[q,
      // sizeB-1-p].
      inIdx[a] = outIdx[b];
      inIdx[b] = sizeB - 1 - outIdx[a];
      std::memcpy(elemPtr(out, outIdx), elemPtr(cur, inIdx), es);
    });
    cur = out;
  }
  return cur;
}

Tensor div(const Tensor& self, double other) {
  if (self.device().type() != DeviceType::CPU) {
    return requireBackend(self.device(), "div").div(self, other);
  }
  Tensor out = empty(self.sizes(), self.scalar_type(), self.device());
  ScalarType dt = self.scalar_type();
  forEachIndex(self.sizes(), [&](const std::vector<int64_t>& idx) {
    double v = loadAsDouble(elemPtr(const_cast<Tensor&>(self), idx), dt);
    storeFromDouble(elemPtr(out, idx), dt, v / other);
  });
  return out;
}

// ---- DLPack interop ----

namespace {

struct ManagedCtx {
  Tensor tensor; // keeps storage alive
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

DLDataType toDLDataType(ScalarType dt) {
  DLDataType d;
  d.lanes = 1;
  switch (dt) {
    case ScalarType::UInt8:
      d.code = kDLUInt;
      d.bits = 8;
      return d;
    case ScalarType::UInt16:
      d.code = kDLUInt;
      d.bits = 16;
      return d;
    case ScalarType::Int32:
      d.code = kDLInt;
      d.bits = 32;
      return d;
    case ScalarType::Int64:
      d.code = kDLInt;
      d.bits = 64;
      return d;
    case ScalarType::Float32:
      d.code = kDLFloat;
      d.bits = 32;
      return d;
    case ScalarType::Float64:
      d.code = kDLFloat;
      d.bits = 64;
      return d;
    case ScalarType::Bool:
      d.code = kDLBool;
      d.bits = 8;
      return d;
  }
  fail("unknown dtype in toDLDataType");
}

ScalarType fromDLDataType(DLDataType d) {
  if (d.lanes != 1) {
    fail("DLPack: only lanes == 1 supported");
  }
  if (d.code == kDLUInt && d.bits == 8) {
    return ScalarType::UInt8;
  }
  if (d.code == kDLUInt && d.bits == 16) {
    return ScalarType::UInt16;
  }
  if (d.code == kDLInt && d.bits == 32) {
    return ScalarType::Int32;
  }
  if (d.code == kDLInt && d.bits == 64) {
    return ScalarType::Int64;
  }
  if (d.code == kDLFloat && d.bits == 32) {
    return ScalarType::Float32;
  }
  if (d.code == kDLFloat && d.bits == 64) {
    return ScalarType::Float64;
  }
  if (d.code == kDLBool && d.bits == 8) {
    return ScalarType::Bool;
  }
  fail("DLPack: unsupported dtype");
}

DLDevice toDLDevice(Device dev) {
  DLDevice d;
  d.device_id = dev.index();
  d.device_type = dev.type() == DeviceType::CUDA ? kDLCUDA : kDLCPU;
  return d;
}

Device fromDLDevice(DLDevice d) {
  switch (d.device_type) {
    case kDLCPU:
    case kDLCUDAHost:
      return Device(DeviceType::CPU, 0);
    case kDLCUDA:
      return Device(DeviceType::CUDA, d.device_id);
    default:
      fail("DLPack: unsupported device type");
  }
}

} // namespace

DLManagedTensor* toDLPack(const Tensor& t) {
  auto* ctx = new ManagedCtx{t, t.sizes(), t.strides()};
  auto* m = new DLManagedTensor();
  m->manager_ctx = ctx;
  m->deleter = [](DLManagedTensor* self) {
    delete static_cast<ManagedCtx*>(self->manager_ctx);
    delete self;
  };
  DLTensor& dl = m->dl_tensor;
  // data points at storage base; element access uses byte_offset.
  dl.data = t.storage().get();
  dl.device = toDLDevice(t.device());
  dl.ndim = static_cast<int32_t>(ctx->shape.size());
  dl.dtype = toDLDataType(t.scalar_type());
  dl.shape = ctx->shape.data();
  dl.strides = ctx->strides.data();
  dl.byte_offset = static_cast<uint64_t>(t.storage_offset() * t.element_size());
  return m;
}

Tensor fromDLPack(DLManagedTensor* managed) {
  const DLTensor& dl = managed->dl_tensor;
  std::vector<int64_t> sizes(dl.shape, dl.shape + dl.ndim);
  std::vector<int64_t> strides;
  if (dl.strides != nullptr) {
    strides.assign(dl.strides, dl.strides + dl.ndim);
  } else {
    strides = contiguousStrides(sizes);
  }
  ScalarType dtype = fromDLDataType(dl.dtype);
  Device device = fromDLDevice(dl.device);

  // Storage shares the producer's buffer; its deleter calls the DLPack deleter
  // exactly once when the last reference dies.
  std::shared_ptr<void> storage(dl.data, [managed](void*) {
    if (managed->deleter != nullptr) {
      managed->deleter(managed);
    }
  });

  int64_t es = elementSize(dtype);
  int64_t offsetElems = es > 0 ? static_cast<int64_t>(dl.byte_offset) / es : 0;
  return Tensor(
      std::move(storage),
      dl.data,
      std::move(sizes),
      std::move(strides),
      dtype,
      device,
      offsetElems);
}

} // namespace facebook::torchcodec::tc
