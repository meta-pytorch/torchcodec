// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <mutex>

#include "CUDACommon.h"
#include "FFMPEGCommon.h"
#include "NVDECCache.h"

#include <cuda_runtime.h> // For cudaGetDevice

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

NVDECCache& NVDECCache::getCache(const StableDevice& device) {
  static NVDECCache cacheInstances[MAX_CUDA_GPUS];
  return cacheInstances[getDeviceIndex(device)];
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

bool NVDECCache::returnDecoder(
    CUVIDEOFORMAT* videoFormat,
    UniqueCUvideodecoder decoder) {
  if (!decoder) {
    return false;
  }

  CacheKey key(videoFormat);
  std::lock_guard<std::mutex> lock(cacheLock_);

  // Evict least recently used entry if at capacity.
  // This is O(MAX_CACHE_SIZE) which should be small enough to be significant.
  if (cache_.size() >= MAX_CACHE_SIZE) {
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
      cache_.size() <= MAX_CACHE_SIZE,
      "Cache size exceeded maximum limit, please report a bug");
  return true;
}

} // namespace facebook::torchcodec
