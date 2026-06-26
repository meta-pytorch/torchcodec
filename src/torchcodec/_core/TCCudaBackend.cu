// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Torch-free CUDA backend for tc::Tensor. This file is compiled into the core
// library ONLY for an ENABLE_CUDA && !ENABLE_TORCH build (the torch build uses
// TCTorchBackend.cpp instead, which routes GPU ops through torch). Its static
// initializer installs:
//   - an allocator hook: malloc for CPU storage, cudaMalloc for CUDA storage
//     (per-frame alloc/free, the approach decord and PyNvVideoCodec also use --
//     no caching allocator is reimplemented).
//   - a CUDA compute backend implementing the ops the GPU decode path needs:
//     copy_ (host<->device / device<->device), zero_, and contiguous (used by
//     the color-conversion output paths). Ops that only arise for float32
//     output, decode-time transforms, or audio (toDtype/div/rot90/cat) require
//     torch and raise a clear error here.
//
// GPU frames are returned as zero-copy CUDA DLPack capsules (tc::toDLPack maps
// the CUDA device); the consumer (cupy / jax / torch) is brought by the user.

#include <cuda_runtime.h>
#include <cstring>
#include <memory>
#include <vector>

#include "TCError.h"
#include "TCTensor.h"

namespace facebook::torchcodec {
namespace {

constexpr int kMaxDims = 8;
struct Dims {
  int64_t v[kMaxDims];
};

// Copies `numel` elements of `elemSize` bytes each from a (possibly strided)
// source into a contiguous (row-major) destination. `srcStrides` are in
// elements. The logical index i is unravelled over `sizes` to find the source
// offset; the destination is contiguous so its offset is just i.
__global__ void stridedToContiguousKernel(
    char* dst,
    const char* src,
    int64_t numel,
    int ndim,
    int elemSize,
    Dims sizes,
    Dims srcStrides) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) {
    return;
  }
  int64_t srcOffElems = 0;
  int64_t rem = i;
  for (int d = ndim - 1; d >= 0; --d) {
    int64_t idx = rem % sizes.v[d];
    rem /= sizes.v[d];
    srcOffElems += idx * srcStrides.v[d];
  }
  const char* s = src + srcOffElems * elemSize;
  char* d = dst + i * elemSize;
  for (int b = 0; b < elemSize; ++b) {
    d[b] = s[b];
  }
}

void launchStridedToContiguous(
    void* dst,
    const void* src,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& srcStrides,
    int elemSize) {
  int ndim = static_cast<int>(sizes.size());
  TC_CHECK(
      ndim <= kMaxDims,
      "torch-free CUDA backend supports up to ",
      kMaxDims,
      " dims, got ",
      ndim);
  int64_t numel = 1;
  Dims s{}, st{};
  for (int d = 0; d < ndim; ++d) {
    s.v[d] = sizes[d];
    st.v[d] = srcStrides[d];
    numel *= sizes[d];
  }
  if (numel == 0) {
    return;
  }
  constexpr int kThreads = 256;
  int64_t blocks = (numel + kThreads - 1) / kThreads;
  stridedToContiguousKernel<<<static_cast<unsigned int>(blocks), kThreads>>>(
      static_cast<char*>(dst),
      static_cast<const char*>(src),
      numel,
      ndim,
      elemSize,
      s,
      st);
  cudaError_t err = cudaGetLastError();
  TC_CHECK(
      err == cudaSuccess,
      "torch-free CUDA strided copy kernel launch failed: ",
      cudaGetErrorString(err));
  err = cudaDeviceSynchronize();
  TC_CHECK(
      err == cudaSuccess,
      "torch-free CUDA strided copy failed: ",
      cudaGetErrorString(err));
}

// ---- Allocator hook (CPU malloc + CUDA cudaMalloc) ----

std::shared_ptr<void>
allocStorage(int64_t numBytes, tc::ScalarType /*dtype*/, tc::Device device) {
  if (numBytes == 0) {
    // Non-null sentinel so defined() is true for empty tensors (mirrors the
    // built-in CPU allocator in TCTensor.cpp).
    return std::shared_ptr<void>(reinterpret_cast<void*>(1), [](void*) {});
  }
  if (device.type() == tc::DeviceType::CPU) {
    void* p = ::operator new(static_cast<size_t>(numBytes));
    return std::shared_ptr<void>(p, [](void* q) { ::operator delete(q); });
  }
  TC_CHECK(
      device.type() == tc::DeviceType::CUDA,
      "torch-free allocator supports only CPU and CUDA devices");
  int index = static_cast<int>(device.index());
  int prevDevice = -1;
  if (index >= 0) {
    cudaGetDevice(&prevDevice);
    cudaSetDevice(index);
  }
  void* p = nullptr;
  cudaError_t err = cudaMalloc(&p, static_cast<size_t>(numBytes));
  if (index >= 0 && prevDevice >= 0) {
    cudaSetDevice(prevDevice);
  }
  TC_CHECK(
      err == cudaSuccess,
      "cudaMalloc of ",
      numBytes,
      " bytes failed: ",
      cudaGetErrorString(err));
  return std::shared_ptr<void>(p, [](void* q) { cudaFree(q); });
}

// ---- CUDA compute backend ----

tc::DeviceBackend makeCudaBackend() {
  tc::DeviceBackend backend;

  backend.copy_ = [](tc::Tensor& dst, const tc::Tensor& src) {
    TC_CHECK(dst.sizes() == src.sizes(), "copy_ requires matching shapes");
    TC_CHECK(
        dst.scalar_type() == src.scalar_type(),
        "torch-free CUDA copy_ requires matching dtypes");
    int elemSize = static_cast<int>(dst.element_size());
    int64_t numel = dst.numel();
    if (numel == 0) {
      return;
    }
    void* dstPtr = dst.mutable_data_ptr();
    const void* srcPtr = src.const_data_ptr();
    if (dst.is_contiguous() && src.is_contiguous()) {
      // Direct byte copy; cudaMemcpyDefault uses UVA to infer H2D/D2H/D2D.
      cudaError_t err = cudaMemcpy(
          dstPtr,
          srcPtr,
          static_cast<size_t>(numel) * elemSize,
          cudaMemcpyDefault);
      TC_CHECK(
          err == cudaSuccess,
          "torch-free CUDA cudaMemcpy failed: ",
          cudaGetErrorString(err));
      return;
    }
    // Strided source into contiguous destination (the only non-contiguous case
    // the decode path produces). A non-contiguous destination would need a
    // scatter and is not exercised.
    TC_CHECK(
        dst.is_contiguous(),
        "torch-free CUDA copy_ requires a contiguous destination");
    launchStridedToContiguous(
        dstPtr, srcPtr, src.sizes(), src.strides(), elemSize);
  };

  backend.zero_ = [](tc::Tensor& self) {
    TC_CHECK(
        self.is_contiguous(),
        "torch-free CUDA zero_ requires a contiguous tensor");
    int64_t n = self.numel();
    if (n == 0) {
      return;
    }
    cudaError_t err = cudaMemset(
        self.mutable_data_ptr(),
        0,
        static_cast<size_t>(n) * self.element_size());
    TC_CHECK(
        err == cudaSuccess,
        "torch-free CUDA cudaMemset failed: ",
        cudaGetErrorString(err));
  };

  backend.contiguous = [](const tc::Tensor& self) -> tc::Tensor {
    if (self.is_contiguous()) {
      return self;
    }
    tc::Tensor out = tc::empty(self.sizes(), self.scalar_type(), self.device());
    launchStridedToContiguous(
        out.mutable_data_ptr(),
        self.const_data_ptr(),
        self.sizes(),
        self.strides(),
        static_cast<int>(self.element_size()));
    return out;
  };

  auto unsupported = [](const char* op) {
    TC_CHECK(
        false,
        "torch-free CUDA backend does not implement ",
        op,
        " (it is only needed for float32 output, decode-time transforms, or "
        "audio, which require torch). Install torch, or use the default uint8 "
        "video output without transforms.");
  };

  backend.toDtype = [unsupported](
                        const tc::Tensor&, tc::ScalarType) -> tc::Tensor {
    unsupported("to(dtype)");
    return {};
  };
  backend.div = [unsupported](const tc::Tensor&, double) -> tc::Tensor {
    unsupported("div");
    return {};
  };
  backend.cat = [unsupported](
                    const std::vector<tc::Tensor>&, int64_t) -> tc::Tensor {
    unsupported("cat");
    return {};
  };
  backend.rot90 =
      [unsupported](const tc::Tensor&, int64_t, int64_t, int64_t) -> tc::Tensor {
    unsupported("rot90");
    return {};
  };

  return backend;
}

struct CudaHookRegistrar {
  CudaHookRegistrar() {
    tc::setAllocator(allocStorage);
    tc::registerDeviceBackend(tc::DeviceType::CUDA, makeCudaBackend());
  }
};

const CudaHookRegistrar gCudaHookRegistrar;

} // namespace
} // namespace facebook::torchcodec
