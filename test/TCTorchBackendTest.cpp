// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Verifies that linking the torch-linked custom-ops library registers tc's
// allocator + CUDA compute backend (TCTorchBackend.cpp), and that tc ops on
// CUDA tensors actually run through torch.

#include "src/torchcodec/_core/StableABICompat.h"
#include "src/torchcodec/_core/TCStableConvert.h"
#include "src/torchcodec/_core/TCTensor.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace facebook::torchcodec::tc {

TEST(TCTorchBackendTest, HooksAreRegistered) {
  // The custom-ops library's static initializer should have run on load.
  EXPECT_TRUE(hasAllocator());
  EXPECT_NE(getDeviceBackend(DeviceType::CUDA), nullptr);
}

TEST(TCTorchBackendTest, CudaRoundTripWithDiv) {
  // CPU source with known values.
  Tensor cpu = empty({4}, ScalarType::Float32);
  auto* p = cpu.mutable_data_ptr<float>();
  p[0] = 0.0f;
  p[1] = 255.0f;
  p[2] = 510.0f;
  p[3] = 51.0f;

  Tensor gpu;
  try {
    // Real decoder usage always runs CUDA work under a device guard; replicate
    // that here so from_blob has the device current.
    StableDeviceGuard guard(/*device_index=*/0);
    // H2D transfer goes through the backend copy_ (via to(device)).
    gpu = to(cpu, Device(DeviceType::CUDA, 0));
  } catch (const std::exception& e) {
    GTEST_SKIP() << "CUDA not available: " << e.what();
  }
  EXPECT_EQ(gpu.device().type(), DeviceType::CUDA);

  // GPU div runs through the torch backend.
  Tensor normalized = div(gpu, 255.0);
  EXPECT_EQ(normalized.device().type(), DeviceType::CUDA);

  // D2H transfer back, then check values.
  Tensor back = to(normalized, Device(DeviceType::CPU));
  EXPECT_EQ(back.device().type(), DeviceType::CPU);
  const auto* bp = back.const_data_ptr<float>();
  EXPECT_NEAR(bp[0], 0.0f, 1e-5);
  EXPECT_NEAR(bp[1], 1.0f, 1e-5);
  EXPECT_NEAR(bp[2], 2.0f, 1e-5);
  EXPECT_NEAR(bp[3], 0.2f, 1e-5);
}

TEST(TCTorchBackendTest, CudaAllocatorUsesTorch) {
  // empty() on CUDA should succeed via the torch allocator hook.
  Tensor t;
  try {
    t = empty({2, 3, 3}, ScalarType::UInt8, Device(DeviceType::CUDA, 0));
  } catch (const std::exception& e) {
    GTEST_SKIP() << "CUDA not available: " << e.what();
  }
  EXPECT_TRUE(t.defined());
  EXPECT_EQ(t.numel(), 18);
  EXPECT_EQ(t.device().type(), DeviceType::CUDA);
}

} // namespace facebook::torchcodec::tc
