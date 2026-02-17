// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "NVDECCacheConfig.h"

#include <atomic>

#include "c10/util/Exception.h"

namespace facebook::torchcodec {

static std::atomic<int> g_nvdecCacheMaxSize{DEFAULT_NVDEC_CACHE_MAX_SIZE};

void setNVDECCacheMaxSize(int size) {
  TORCH_CHECK(size >= 0, "NVDEC cache size must be non-negative, got ", size);
  g_nvdecCacheMaxSize.store(size);
}

int getNVDECCacheMaxSize() {
  return g_nvdecCacheMaxSize.load();
}

} // namespace facebook::torchcodec
