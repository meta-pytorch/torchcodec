// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "NVDECCacheConfig.h"

#include <atomic>
#include <mutex>

#include "c10/util/Exception.h"

#ifdef USE_CUDA
#include "NVDECCache.h"
#endif

namespace facebook::torchcodec {

static std::atomic<int> g_nvdecCacheMaxSize{DEFAULT_NVDEC_CACHE_MAX_SIZE};
// This mutex serializes setNVDECCacheMaxSize() calls so that the atomic store
// and the subsequent cache eviction happen as one unit. getNVDECCacheMaxSize()
// intentionally reads the atomic without this mutex: callers like
// returnDecoder() may briefly see a stale value during an ongoing
// setNVDECCacheMaxSize(), which is acceptable because the eviction pass will
// correct the cache size momentarily.
static std::mutex g_nvdecCacheMaxSizeMutex;

void setNVDECCacheMaxSize(int size) {
  TORCH_CHECK(size >= 0, "NVDEC cache size must be non-negative, got ", size);
  std::lock_guard<std::mutex> lock(g_nvdecCacheMaxSizeMutex);
  g_nvdecCacheMaxSize.store(size);
#ifdef USE_CUDA
  NVDECCache::evictExcessEntriesAcrossDevices(size);
#endif
}

int getNVDECCacheMaxSize() {
  return g_nvdecCacheMaxSize.load();
}

} // namespace facebook::torchcodec
