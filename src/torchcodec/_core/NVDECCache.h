// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <map>
#include <memory>
#include <mutex>

#include <cuda.h>

#include "CUDACommon.h"
#include "EvictionPolicies.h"
#include "NVCUVIDRuntimeLoader.h"
#include "StableABICompat.h"
#include "nvcuvid_include/cuviddec.h"
#include "nvcuvid_include/nvcuvid.h"

namespace facebook::torchcodec {

// This file implements a cache for NVDEC decoders.
// TODONVDEC P3: Consider merging this with Cache.h. The main difference is that
// this NVDEC Cache involves a cache key (the decoder parameters).

struct CUvideoDecoderDeleter {
  void operator()(CUvideodecoder* decoderPtr) const {
    if (decoderPtr && *decoderPtr) {
      cuvidDestroyDecoder(*decoderPtr);
      delete decoderPtr;
    }
  }
};

using UniqueCUvideodecoder =
    std::unique_ptr<CUvideodecoder, CUvideoDecoderDeleter>;

// Cache entry that holds a decoder and tracks whether it's currently in use.
struct CacheEntry {
  UniqueCUvideodecoder decoder;
  bool inUse;

  CacheEntry(UniqueCUvideodecoder dec, bool used)
      : decoder(std::move(dec)), inUse(used) {}
};

// Cache key struct: a decoder can be reused and taken from the cache only if
// all these parameters match.
struct NVDECCacheKey {
  cudaVideoCodec codecType;
  uint32_t width;
  uint32_t height;
  cudaVideoChromaFormat chromaFormat;
  uint32_t bitDepthLumaMinus8;
  uint8_t numDecodeSurfaces;

  NVDECCacheKey() = delete;

  explicit NVDECCacheKey(CUVIDEOFORMAT* videoFormat)
      : codecType(videoFormat->codec),
        width(videoFormat->coded_width),
        height(videoFormat->coded_height),
        chromaFormat(videoFormat->chroma_format),
        bitDepthLumaMinus8(videoFormat->bit_depth_luma_minus8),
        numDecodeSurfaces(videoFormat->min_num_decode_surfaces) {}

  NVDECCacheKey(const NVDECCacheKey&) = default;
  NVDECCacheKey& operator=(const NVDECCacheKey&) = default;

  bool operator<(const NVDECCacheKey& other) const {
    return std::tie(
               codecType,
               width,
               height,
               chromaFormat,
               bitDepthLumaMinus8,
               numDecodeSurfaces) <
        std::tie(
               other.codecType,
               other.width,
               other.height,
               other.chromaFormat,
               other.bitDepthLumaMinus8,
               other.numDecodeSurfaces);
  }
};

// Type aliases for cache types
using NVDECCacheMap = std::multimap<NVDECCacheKey, CacheEntry>;
using NVDECCacheIterator = NVDECCacheMap::iterator;

// Default eviction policy - can be changed at compile time
using DefaultNVDECEvictionPolicy = LRUEvictionPolicy<NVDECCacheIterator>;

// A per-device cache for NVDEC decoders. There is one instance of this class
// per GPU device, and it is accessed through the static getCache() method.
// The cache supports multiple decoders with the same parameters, tracking
// which ones are currently in use.
//
// Template parameter EvictionPolicy controls what happens when the cache is
// full and a new decoder needs to be added. Available policies:
//   - LRUEvictionPolicy: Evicts least recently used entry (default)
//   - FIFOEvictionPolicy: Evicts oldest entry
//   - NoEvictionPolicy: Rejects new entries when full (original behavior)
template <typename EvictionPolicy = DefaultNVDECEvictionPolicy>
class NVDECCacheImpl {
 public:
  // Max number of cached decoders, per device
  static constexpr int MAX_CACHE_SIZE = 20;

  static NVDECCacheImpl& getCache(const StableDevice& device) {
    static NVDECCacheImpl cacheInstances[MAX_CUDA_GPUS];
    return cacheInstances[getDeviceIndex(device)];
  }

  // Get decoder from cache - returns nullptr if none available.
  // The returned decoder is marked as "in use" until returned via
  // returnDecoder.
  UniqueCUvideodecoder getDecoder(CUVIDEOFORMAT* videoFormat) {
    NVDECCacheKey key(videoFormat);
    std::lock_guard<std::mutex> lock(cacheLock_);

    // Find all entries with matching key and look for one not in use
    auto range = cache_.equal_range(key);
    for (auto it = range.first; it != range.second; ++it) {
      if (!it->second.inUse) {
        // Take ownership of the decoder and remove the entry from cache
        auto decoder = std::move(it->second.decoder);
        evictionPolicy_.onRemove(it);
        cache_.erase(it);
        return decoder;
      }
    }

    return nullptr;
  }

  // Return decoder to cache - marks the decoder as not in use.
  // Returns true if the decoder was successfully returned to cache.
  bool returnDecoder(CUVIDEOFORMAT* videoFormat, UniqueCUvideodecoder decoder) {
    if (!decoder) {
      return false;
    }

    NVDECCacheKey key(videoFormat);
    std::lock_guard<std::mutex> lock(cacheLock_);

    // If at capacity, try to evict using the policy
    while (cache_.size() >= MAX_CACHE_SIZE && !evictionPolicy_.empty()) {
      auto victimIt = evictionPolicy_.selectForEviction();
      evictionPolicy_.onRemove(victimIt);
      cache_.erase(victimIt);
    }

    // If still at capacity (policy couldn't evict), reject the decoder
    if (cache_.size() >= MAX_CACHE_SIZE) {
      return false;
    }

    // Add the decoder back to cache as not in use
    auto it = cache_.emplace(key, CacheEntry(std::move(decoder), false));
    evictionPolicy_.onInsert(it);
    return true;
  }

 private:
  NVDECCacheImpl() = default;
  ~NVDECCacheImpl() = default;

  NVDECCacheMap cache_;
  EvictionPolicy evictionPolicy_;
  std::mutex cacheLock_;
};

// Type alias for the default cache implementation
// To change the eviction policy globally, modify DefaultNVDECEvictionPolicy
// or use a different instantiation:
//   using NVDECCache = NVDECCacheImpl<NoEvictionPolicy<NVDECCacheIterator>>;
using NVDECCache = NVDECCacheImpl<DefaultNVDECEvictionPolicy>;

} // namespace facebook::torchcodec
