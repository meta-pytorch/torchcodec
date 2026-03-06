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

#include "NVCUVIDRuntimeLoader.h"
#include "NVDECCacheConfig.h"
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

struct CacheEntry {
  UniqueCUvideodecoder decoder;
  uint64_t lastUsed; // LRU timestamp

  CacheEntry(UniqueCUvideodecoder dec, uint64_t ts)
      : decoder(std::move(dec)), lastUsed(ts) {}
};

// A per-device LRU cache for NVDEC decoders. There is one instance of this
// class per GPU device, and it is accessed through the static getCache()
// method.  The cache supports multiple decoders with the same parameters.
class NVDECCache {
 public:
  static NVDECCache& getCache(const StableDevice& device);

  // Get decoder from cache - returns nullptr if none available.
  UniqueCUvideodecoder getDecoder(CUVIDEOFORMAT* videoFormat);

  // Return decoder to cache using LRU eviction.
  void returnDecoder(CUVIDEOFORMAT* videoFormat, UniqueCUvideodecoder decoder);

  // Iterates all per-device cache instances and evicts LRU entries until each
  // cache's size is at most capacity. Called from setNVDECCacheCapacity().
  static void evictExcessEntriesAcrossDevices(int capacity);

  // Returns the number of entries in the cache for a given device index.
  static int getCacheSizeForDevice(int device_index);

 private:
  // Cache key struct: a decoder can be reused and taken from the cache only if
  // all these parameters match.
  struct CacheKey {
    cudaVideoCodec codecType;
    uint32_t width;
    uint32_t height;
    cudaVideoChromaFormat chromaFormat;
    uint32_t bitDepthLumaMinus8;
    uint8_t numDecodeSurfaces;

    CacheKey() = delete;

    explicit CacheKey(CUVIDEOFORMAT* videoFormat) {
      STD_TORCH_CHECK(videoFormat != nullptr, "videoFormat must not be null");
      codecType = videoFormat->codec;
      width = videoFormat->coded_width;
      height = videoFormat->coded_height;
      chromaFormat = videoFormat->chroma_format;
      bitDepthLumaMinus8 = videoFormat->bit_depth_luma_minus8;
      numDecodeSurfaces = videoFormat->min_num_decode_surfaces;
    }

    CacheKey(const CacheKey&) = default;
    CacheKey& operator=(const CacheKey&) = default;

    bool operator<(const CacheKey& other) const {
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

  NVDECCache() = default;
  ~NVDECCache() = default;

  void evictLRUEntry();

  static NVDECCache* getCacheInstances();

  std::multimap<CacheKey, CacheEntry> cache_;
  std::mutex cacheLock_;
  uint64_t lastUsedCounter_ = 0;
};

} // namespace facebook::torchcodec
