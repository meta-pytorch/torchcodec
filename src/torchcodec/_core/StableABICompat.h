// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/DeviceType.h>

#include <stdexcept>
#include <string>

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
using StableDeviceIndex = torch::stable::accelerator::DeviceIndex;

// DeviceGuard for CUDA context management
using StableDeviceGuard = torch::stable::accelerator::DeviceGuard;

// Device type constants
constexpr auto kStableCPU = torch::headeronly::DeviceType::CPU;
constexpr auto kStableCUDA = torch::headeronly::DeviceType::CUDA;

} // namespace facebook::torchcodec
