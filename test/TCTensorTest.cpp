// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/TCTensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace facebook::torchcodec::tc {

TEST(TCTensorTest, EmptyAndMetadata) {
  Tensor t = empty({2, 3, 4}, ScalarType::UInt8);
  EXPECT_TRUE(t.defined());
  EXPECT_EQ(t.dim(), 3);
  EXPECT_EQ(t.numel(), 24);
  EXPECT_EQ(t.element_size(), 1);
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_EQ(t.strides(), (std::vector<int64_t>{12, 4, 1}));
  EXPECT_EQ(t.device().type(), DeviceType::CPU);
}

TEST(TCTensorTest, FullScalar) {
  Tensor s = full({}, 3.5, ScalarType::Float64); // 0-dim scalar
  EXPECT_EQ(s.dim(), 0);
  EXPECT_EQ(s.numel(), 1);
  EXPECT_EQ(*s.const_data_ptr<double>(), 3.5);
}

TEST(TCTensorTest, FromBlobInvokesDeleter) {
  bool deleted = false;
  auto* buf = new float[6]{1, 2, 3, 4, 5, 6};
  {
    Tensor t = from_blob(
        buf, {2, 3}, {3, 1}, ScalarType::Float32, Device{},
        [&deleted](void* p) {
          deleted = true;
          delete[] static_cast<float*>(p);
        });
    EXPECT_EQ(t.const_data_ptr<float>()[4], 5.0f);
    EXPECT_FALSE(deleted);
  }
  EXPECT_TRUE(deleted); // deleter ran when last reference died
}

TEST(TCTensorTest, CopyAndZero) {
  Tensor a = empty({2, 2}, ScalarType::Int32);
  auto* ap = a.mutable_data_ptr<int32_t>();
  ap[0] = 10; ap[1] = 20; ap[2] = 30; ap[3] = 40;
  Tensor b = empty({2, 2}, ScalarType::Int32);
  copy_(b, a);
  EXPECT_EQ(b.const_data_ptr<int32_t>()[3], 40);
  zero_(b);
  EXPECT_EQ(b.const_data_ptr<int32_t>()[0], 0);
  EXPECT_EQ(b.const_data_ptr<int32_t>()[3], 0);
}

TEST(TCTensorTest, ToDtypeAndDiv) {
  Tensor a = empty({4}, ScalarType::UInt8);
  auto* ap = a.mutable_data_ptr<uint8_t>();
  ap[0] = 0; ap[1] = 127; ap[2] = 200; ap[3] = 255;
  Tensor f = to(a, ScalarType::Float32);
  EXPECT_EQ(f.scalar_type(), ScalarType::Float32);
  EXPECT_EQ(f.const_data_ptr<float>()[2], 200.0f);
  Tensor d = div(f, 255.0);
  EXPECT_NEAR(d.const_data_ptr<float>()[3], 1.0f, 1e-6);
}

TEST(TCTensorTest, ViewsSelectNarrowPermuteContiguous) {
  // 3x4 contiguous, values 0..11
  Tensor a = empty({3, 4}, ScalarType::Int32);
  auto* ap = a.mutable_data_ptr<int32_t>();
  for (int i = 0; i < 12; ++i) {
    ap[i] = i;
  }

  Tensor row = selectRow(a, 1); // [4,5,6,7]
  EXPECT_EQ(row.dim(), 1);
  EXPECT_EQ(row.const_data_ptr<int32_t>()[0], 4);
  EXPECT_EQ(row.const_data_ptr<int32_t>()[3], 7);

  Tensor n = narrow(a, 1, 1, 2); // columns 1..2 -> shape [3,2]
  EXPECT_EQ(n.sizes(), (std::vector<int64_t>{3, 2}));
  // element [0,0] should be original [0,1] == 1
  EXPECT_EQ(
      *reinterpret_cast<int32_t*>(static_cast<char*>(n.mutable_data_ptr())), 1);

  Tensor p = permute(a, {1, 0}); // transpose -> [4,3], non-contiguous
  EXPECT_EQ(p.sizes(), (std::vector<int64_t>{4, 3}));
  EXPECT_FALSE(p.is_contiguous());
  Tensor pc = contiguous(p);
  EXPECT_TRUE(pc.is_contiguous());
  // pc[0] = column 0 of a = [0,4,8]
  EXPECT_EQ(pc.const_data_ptr<int32_t>()[0], 0);
  EXPECT_EQ(pc.const_data_ptr<int32_t>()[1], 4);
  EXPECT_EQ(pc.const_data_ptr<int32_t>()[2], 8);
}

TEST(TCTensorTest, Cat) {
  Tensor a = full({2, 2}, 1.0, ScalarType::Float32);
  Tensor b = full({2, 3}, 2.0, ScalarType::Float32);
  Tensor c = cat({a, b}, 1); // -> [2,5]
  EXPECT_EQ(c.sizes(), (std::vector<int64_t>{2, 5}));
  EXPECT_EQ(c.const_data_ptr<float>()[0], 1.0f); // from a
  EXPECT_EQ(c.const_data_ptr<float>()[2], 2.0f); // from b
}

TEST(TCTensorTest, Rot90MatchesNumpy) {
  // 2x3 matrix [[0,1,2],[3,4,5]]
  Tensor a = empty({2, 3}, ScalarType::Int32);
  auto* ap = a.mutable_data_ptr<int32_t>();
  for (int i = 0; i < 6; ++i) {
    ap[i] = i;
  }
  Tensor r = rot90(a, 1, 0, 1); // counter-clockwise -> shape [3,2]
  EXPECT_EQ(r.sizes(), (std::vector<int64_t>{3, 2}));
  // numpy.rot90([[0,1,2],[3,4,5]]) = [[2,5],[1,4],[0,3]]
  const auto* rp = r.const_data_ptr<int32_t>();
  EXPECT_EQ(rp[0], 2);
  EXPECT_EQ(rp[1], 5);
  EXPECT_EQ(rp[2], 1);
  EXPECT_EQ(rp[3], 4);
  EXPECT_EQ(rp[4], 0);
  EXPECT_EQ(rp[5], 3);
}

TEST(TCTensorTest, DLPackRoundtripIsZeroCopy) {
  Tensor a = empty({2, 3}, ScalarType::Float32);
  auto* ap = a.mutable_data_ptr<float>();
  for (int i = 0; i < 6; ++i) {
    ap[i] = static_cast<float>(i) + 0.5f;
  }

  DLManagedTensor* m = toDLPack(a);
  EXPECT_EQ(m->dl_tensor.ndim, 2);
  EXPECT_EQ(m->dl_tensor.shape[0], 2);
  EXPECT_EQ(m->dl_tensor.shape[1], 3);
  EXPECT_EQ(m->dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(m->dl_tensor.dtype.bits, 32);
  EXPECT_EQ(m->dl_tensor.device.device_type, kDLCPU);

  Tensor b = fromDLPack(m); // adopts m
  EXPECT_EQ(b.sizes(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(b.scalar_type(), ScalarType::Float32);
  // zero-copy: same underlying buffer
  EXPECT_EQ(b.const_data_ptr<float>(), a.const_data_ptr<float>());
  EXPECT_EQ(b.const_data_ptr<float>()[5], 5.5f);
}

TEST(TCTensorTest, AllocatorHookCpuDefault) {
  // No allocator installed: CPU works via malloc, non-CPU throws.
  EXPECT_FALSE(hasAllocator());
  Tensor c = empty({2, 2}, ScalarType::Float32);
  EXPECT_TRUE(c.defined());
  EXPECT_THROW(
      empty({2, 2}, ScalarType::Float32, Device(DeviceType::CUDA, 0)),
      std::runtime_error);
}

TEST(TCTensorTest, AllocatorHookIsUsed) {
  // Install an allocator that records calls and serves host memory (we label
  // it CUDA but never dereference it as device memory).
  int calls = 0;
  Device sawDevice;
  int64_t sawBytes = 0;
  setAllocator([&](int64_t numBytes, ScalarType, Device device)
                   -> std::shared_ptr<void> {
    ++calls;
    sawDevice = device;
    sawBytes = numBytes;
    void* p = ::operator new(static_cast<size_t>(numBytes));
    return std::shared_ptr<void>(p, [](void* q) { ::operator delete(q); });
  });
  EXPECT_TRUE(hasAllocator());

  Tensor t = empty({2, 3}, ScalarType::Float32, Device(DeviceType::CUDA, 1));
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(sawBytes, 2 * 3 * 4); // 6 float32 elements
  EXPECT_EQ(sawDevice.type(), DeviceType::CUDA);
  EXPECT_EQ(sawDevice.index(), 1);
  EXPECT_EQ(t.device().type(), DeviceType::CUDA);
  EXPECT_TRUE(t.defined());

  // Clean up global state so other tests see the default (no allocator).
  setAllocator(nullptr);
  EXPECT_FALSE(hasAllocator());
}

TEST(TCTensorTest, DLPackKeepsStorageAlive) {
  // Export, drop the original tensor, ensure data still valid via DLPack ctx.
  DLManagedTensor* m = nullptr;
  {
    Tensor a = empty({4}, ScalarType::Int32);
    auto* ap = a.mutable_data_ptr<int32_t>();
    for (int i = 0; i < 4; ++i) {
      ap[i] = i * 11;
    }
    m = toDLPack(a);
  } // a destroyed, but DLPack ctx holds a reference
  Tensor b = fromDLPack(m);
  EXPECT_EQ(b.const_data_ptr<int32_t>()[3], 33);
}

} // namespace facebook::torchcodec::tc
