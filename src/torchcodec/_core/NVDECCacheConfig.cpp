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
#include "CUDACommon.h"
#include "NVDECCache.h"
#endif

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

void setNVDECCacheCapacity(int capacity) {
  TORCH_CHECK(
      capacity >= 0,
      "NVDEC cache capacity must be non-negative, got ",
      capacity);
  std::lock_guard<std::mutex> lock(g_nvdecCacheCapacityMutex);
  g_nvdecCacheCapacity.store(capacity);
#ifdef USE_CUDA
  NVDECCache::evictExcessEntriesAcrossDevices(capacity);
#endif
}

int getNVDECCacheCapacity() {
  return g_nvdecCacheCapacity.load();
}

int getNVDECCacheSize([[maybe_unused]] int device_index) {
#ifdef USE_CUDA
  TORCH_CHECK(
      device_index >= 0 && device_index < MAX_CUDA_GPUS,
      "device_index must be between 0 and ",
      MAX_CUDA_GPUS - 1,
      ", got ",
      device_index);
  return NVDECCache::getCacheSizeForDevice(device_index);
#else
  return 0;
#endif
}

} // namespace facebook::torchcodec
