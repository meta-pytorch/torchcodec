// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <mutex>

#include "CUDACommon.h"
#include "FFMPEGCommon.h"
#include "NVDECCache.h"
#include "NVDECCacheConfig.h"

#include <cuda_runtime.h> // For cudaGetDevice

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

NVDECCache* NVDECCache::getCacheInstances() {
  static NVDECCache cacheInstances[MAX_CUDA_GPUS];
  return cacheInstances;
}

NVDECCache& NVDECCache::getCache(const StableDevice& device) {
  return getCacheInstances()[getDeviceIndex(device)];
}

UniqueCUvideodecoder NVDECCache::getDecoder(CUVIDEOFORMAT* videoFormat) {
  CacheKey key(videoFormat);
  std::lock_guard<std::mutex> lock(cacheLock_);

  // Find an entry with matching key
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    auto decoder = std::move(it->second.decoder);
    cache_.erase(it);
    return decoder;
  }

  return nullptr;
}

// Evicts the least-recently-used entry from cache_.
// Caller must hold cacheLock_!!!
void NVDECCache::evictLRUEntry() {
  if (cache_.empty()) {
    return;
  }
  auto victim = cache_.begin();
  for (auto it = cache_.begin(); it != cache_.end(); ++it) {
    if (it->second.lastUsed < victim->second.lastUsed) {
      victim = it;
    }
  }
  cache_.erase(victim);
}

void NVDECCache::returnDecoder(
    CUVIDEOFORMAT* videoFormat,
    UniqueCUvideodecoder decoder) {
  STD_TORCH_CHECK(decoder != nullptr, "decoder must not be null");

  CacheKey key(videoFormat);
  std::lock_guard<std::mutex> lock(cacheLock_);

  int capacity = getNVDECCacheCapacity();
  if (capacity <= 0) {
    return;
  }

  // Evict least recently used entries until under capacity.
  // This search is O(capacity), which is supposed to be small,
  // so linear vs constant search overhead is expected to be negligible.
  while (cache_.size() >= static_cast<size_t>(capacity)) {
    evictLRUEntry();
  }

  // Add the decoder back to cache
  cache_.emplace(key, CacheEntry(std::move(decoder), lastUsedCounter_++));

  STD_TORCH_CHECK(
      cache_.size() <= static_cast<size_t>(capacity),
      "Cache size exceeded capacity, please report a bug");
}

void NVDECCache::evictExcessEntriesAcrossDevices(int capacity) {
  NVDECCache* instances = getCacheInstances();
  for (int i = 0; i < MAX_CUDA_GPUS; ++i) {
    std::lock_guard<std::mutex> lock(instances[i].cacheLock_);
    while (instances[i].cache_.size() > static_cast<size_t>(capacity)) {
      instances[i].evictLRUEntry();
    }
  }
}

int NVDECCache::getCacheSizeForDevice(int device_index) {
  NVDECCache* instances = getCacheInstances();
  std::lock_guard<std::mutex> lock(instances[device_index].cacheLock_);
  return static_cast<int>(instances[device_index].cache_.size());
}

} // namespace facebook::torchcodec
