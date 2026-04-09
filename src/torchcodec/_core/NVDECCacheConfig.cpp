// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "NVDECCacheConfig.h"

#include <atomic>
#include <mutex>

#include "c10/util/Exception.h"

namespace facebook::torchcodec {

static std::atomic<int> g_nvdecCacheCapacity{DEFAULT_NVDEC_CACHE_CAPACITY};
// This mutex serializes setNVDECCacheCapacity() calls so that the atomic store
// and the subsequent cache eviction happen as one unit. getNVDECCacheCapacity()
// intentionally reads the atomic without this mutex: callers like
// returnDecoder() may briefly see a stale value during an ongoing
// setNVDECCacheCapacity(), which is acceptable because the worst case is a
// single decoder being added back to the cache after eviction. That entry will
// be consumed by a subsequent getDecoder() call or evicted by a future
// returnDecoder() or setNVDECCacheCapacity() call.
static std::mutex g_nvdecCacheCapacityMutex;

// Callbacks registered by the CUDA library at load time.
static EvictCacheEntriesFn g_evictFn = nullptr;
static GetCacheSizeFn g_getCacheSizeFn = nullptr;

void registerNVDECCacheCallbacks(
    EvictCacheEntriesFn evict,
    GetCacheSizeFn getSize) {
  g_evictFn = evict;
  g_getCacheSizeFn = getSize;
}

void setNVDECCacheCapacity(int capacity) {
  TORCH_CHECK(
      capacity >= 0,
      "NVDEC cache capacity must be non-negative, got ",
      capacity);
  std::lock_guard<std::mutex> lock(g_nvdecCacheCapacityMutex);
  g_nvdecCacheCapacity.store(capacity);
  if (g_evictFn) {
    g_evictFn(capacity);
  }
}

int getNVDECCacheCapacity() {
  return g_nvdecCacheCapacity.load();
}

int getNVDECCacheSize([[maybe_unused]] int device_index) {
  if (g_getCacheSizeFn) {
    return g_getCacheSizeFn(device_index);
  }
  return 0;
}

} // namespace facebook::torchcodec
