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

NVDECCache* NVDECCache::get_cache_instances() {
  // Intentionally leaked to avoid calling into CUDA/NVCUVID during static
  // destruction, when the CUDA runtime may already be torn down.
  static NVDECCache* cache_instances = new NVDECCache[MAX_CUDA_GPUS];
  return cache_instances;
}

NVDECCache& NVDECCache::get_cache(const StableDevice& device) {
  return get_cache_instances()[get_device_index(device)];
}

UniqueCUvideodecoder NVDECCache::get_decoder(
    CUVIDEOFORMAT* video_format,
    cudaVideoSurfaceFormat surface_format) {
  CacheKey key(video_format, surface_format);
  std::lock_guard<std::mutex> lock(cache_lock_);

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
void NVDECCache::evict_lru_entry() {
  if (cache_.empty()) {
    return;
  }
  auto victim = cache_.begin();
  for (auto it = cache_.begin(); it != cache_.end(); ++it) {
    if (it->second.last_used < victim->second.last_used) {
      victim = it;
    }
  }
  cache_.erase(victim);
}

void NVDECCache::return_decoder(
    CUVIDEOFORMAT* video_format,
    cudaVideoSurfaceFormat surface_format,
    UniqueCUvideodecoder decoder) {
  STD_TORCH_CHECK(decoder != nullptr, "decoder must not be null");

  CacheKey key(video_format, surface_format);
  std::lock_guard<std::mutex> lock(cache_lock_);

  int capacity = get_nvdec_cache_capacity();
  if (capacity <= 0) {
    return;
  }

  // Evict least recently used entries until under capacity.
  // This search is O(capacity), which is supposed to be small,
  // so linear vs constant search overhead is expected to be negligible.
  while (cache_.size() >= static_cast<size_t>(capacity)) {
    evict_lru_entry();
  }

  // Add the decoder back to cache
  cache_.emplace(key, CacheEntry(std::move(decoder), last_used_counter_++));

  STD_TORCH_CHECK(
      cache_.size() <= static_cast<size_t>(capacity),
      "Cache size exceeded capacity, please report a bug");
}

void NVDECCache::evict_excess_entries_across_devices(int capacity) {
  NVDECCache* instances = get_cache_instances();
  for (int i = 0; i < MAX_CUDA_GPUS; ++i) {
    std::lock_guard<std::mutex> lock(instances[i].cache_lock_);
    while (instances[i].cache_.size() > static_cast<size_t>(capacity)) {
      instances[i].evict_lru_entry();
    }
  }
}

int NVDECCache::get_cache_size_for_device(int device_index) {
  NVDECCache* instances = get_cache_instances();
  std::lock_guard<std::mutex> lock(instances[device_index].cache_lock_);
  return static_cast<int>(instances[device_index].cache_.size());
}

} // namespace facebook::torchcodec
