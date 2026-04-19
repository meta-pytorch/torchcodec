// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// This header is intentionally CUDA-free so it can be included from
// custom_ops.cpp which is compiled without CUDA headers.

namespace facebook::torchcodec {

// Default capacity of the per-device NVDEC decoder cache.
// capacity == maximum number of cached instances allowed.
constexpr int DEFAULT_NVDEC_CACHE_CAPACITY = 20;

// Set the capacity of the per-device NVDEC decoder cache.
// capacity must be non-negative.
void setNVDECCacheCapacity(int capacity);

// Get the current capacity of the per-device NVDEC decoder cache.
int getNVDECCacheCapacity();

// Get the current number of entries in the NVDEC decoder cache for a device.
// This is currently only used for tests, and not publicly exposed.
// TODO expose it?
int getNVDECCacheSize(int device_index);

} // namespace facebook::torchcodec
