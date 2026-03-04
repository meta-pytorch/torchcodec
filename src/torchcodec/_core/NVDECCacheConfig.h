// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// This header is intentionally CUDA-free so it can be included from
// custom_ops.cpp which is compiled without CUDA headers.

namespace facebook::torchcodec {

// Default max number of cached NVDEC decoders per device.
constexpr int DEFAULT_NVDEC_CACHE_MAX_SIZE = 20;

// Set the maximum number of NVDEC decoders cached per device.
// size must be non-negative.
void setNVDECCacheMaxSize(int size);

// Get the current maximum number of NVDEC decoders cached per device.
int getNVDECCacheMaxSize();

} // namespace facebook::torchcodec
