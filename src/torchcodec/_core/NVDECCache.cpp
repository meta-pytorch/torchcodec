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

void NVDECCache::returnDecoder(
    CUVIDEOFORMAT* videoFormat,
    UniqueCUvideodecoder decoder) {
  STD_TORCH_CHECK(decoder != nullptr, "decoder must not be null");

  CacheKey key(videoFormat);
  std::lock_guard<std::mutex> lock(cacheLock_);

  if (getNVDECCacheMaxSize() <= 0) {
    return;
  }

  // Evict least recently used entry if at capacity.
  // This search is O(MAX_CACHE_SIZE), MAX_CACHE_SIZE is supposed to be small,
  // so linear vs constant search overhead is expected to be negligible.
  if (cache_.size() >= static_cast<size_t>(getNVDECCacheMaxSize())) {
    auto victim = cache_.begin();
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
      if (it->second.lastUsed < victim->second.lastUsed) {
        victim = it;
      }
    }
    cache_.erase(victim);
  }

  // Add the decoder back to cache
  cache_.emplace(key, CacheEntry(std::move(decoder), lastUsedCounter_++));

  STD_TORCH_CHECK(
      cache_.size() <= static_cast<size_t>(getNVDECCacheMaxSize()),
      "Cache size exceeded maximum limit, please report a bug");
}

size_t NVDECCache::size() {
  std::lock_guard<std::mutex> lock(cacheLock_);
  return cache_.size();
}

int NVDECCache::getMaxCurrentSizeAcrossDevices() {
  NVDECCache* instances = getCacheInstances();
  int maxSize = 0;
  for (int i = 0; i < MAX_CUDA_GPUS; ++i) {
    int s = static_cast<int>(instances[i].size());
    if (s > maxSize) {
      maxSize = s;
    }
  }
  return maxSize;
}

} // namespace facebook::torchcodec
