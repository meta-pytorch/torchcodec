// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/types.h>
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

NVDECCache& NVDECCache::getCache(const torch::Device& device) {
  static NVDECCache cacheInstances[MAX_CUDA_GPUS];
  return cacheInstances[getDeviceIndex(device)];
}

// Simple ID allocator for decoders - not strictly necessary, but useful for
// reconfiguring decoders.
uint32_t NVDECCache::allocDecoderId() {
  std::lock_guard<std::mutex> lock(cacheLock_);
  return nextId++;
}

// Register a decoder ID with its maximum width and height.
bool NVDECCache::registerDecoderId(uint32_t decoderId,
    uint32_t ulMaxWidth,
    uint32_t ulMaxHeight) {
  std::lock_guard<std::mutex> lock(cacheLock_);
  auto it = id_context_map_.find(decoderId);
  if (it != id_context_map_.end()) {
    // Already registered
    return false;
  }
  id_context_map_[decoderId] = DecoderMaxWHContext{decoderId, ulMaxWidth, ulMaxHeight};
  return true;

}


CUvideodecoderCache NVDECCache::getDecoder(CUVIDEOFORMAT* videoFormat) {
  NVDECCacheType cache_type = NVDECCacheType::Create;
  CacheKey key(videoFormat);
  std::lock_guard<std::mutex> lock(cacheLock_);

  auto it = cache_.find(key);
  if (it != cache_.end() && it->second.size() > 0) {
    auto it2 = context_cache_.find(key);
    TORCH_CHECK(
      it2 != context_cache_.end(),
      "Decoder context cache inconsistency detected."
    );
    TORCH_CHECK(
      it->second.size() == it2->second.size(),
      "Size of cache_[key] and context_cache_[key] do not match."
    );

    // We first check if the cached decoder can be reused as is. If the number of
    // surfaces allocated for the cached decoder is equal to the requested number of
    // surfaces, and the coded dimensions also match, then we can reuse it directly.
    // Otherwise, we need to reconfigure it.
    for (auto bg = it2->second.begin(), ed = it2->second.end(); bg != ed; ++bg) {
      const auto& context = *bg;
      if (
        context.first.numDecodeSurfaces == videoFormat->min_num_decode_surfaces &&
        context.first.coded_width == videoFormat->coded_width &&
        context.first.coded_height == videoFormat->coded_height
      ) {
        cache_type = NVDECCacheType::Reuse;

        // Delete the selected decoder from the cache lists
        auto dist = std::distance(it2->second.begin(), bg);
        auto decoder_it = it->second.begin();
        std::advance(decoder_it, dist);
        auto decoder = std::move(*decoder_it);;
        it->second.erase(decoder_it);
        it2->second.erase(bg);
        return std::make_tuple(cache_type, std::move(decoder), context.second.decoderID);
      }
    }

    for (auto bg = it2->second.begin(), ed = it2->second.end(); bg != ed; ++bg) {
      const auto& context = *bg;
      if (
        context.second.ulMaxWidth >= videoFormat->coded_width &&
        context.second.ulMaxHeight >= videoFormat->coded_height
      ) {
        cache_type = NVDECCacheType::Reconfig;

        // Delete the selected decoder from the cache lists
        auto dist = std::distance(it2->second.begin(), bg);
        auto decoder_it = it->second.begin();
        std::advance(decoder_it, dist);
        auto decoder = std::move(*decoder_it);
        it->second.erase(decoder_it);
        it2->second.erase(bg);
        return std::make_tuple(cache_type, std::move(decoder), context.second.decoderID);
      }
    }
  }

  return std::make_tuple(cache_type, nullptr, 0);
}

bool NVDECCache::returnDecoder(
    CUVIDEOFORMAT* videoFormat,
    UniqueCUvideodecoder decoder,
    uint32_t decoderId) {
  if (!decoder) {
    return false;
  }

  CacheKey key(videoFormat);
  std::lock_guard<std::mutex> lock(cacheLock_);

  uint32_t current_cache_size = 0;
  for (const auto& pair : cache_) {
    current_cache_size += pair.second.size();
  }

  if (current_cache_size >= MAX_CACHE_SIZE) {
    return false;
  }

  auto it = id_context_map_.find(decoderId);
  TORCH_CHECK(
    it != id_context_map_.end(),
    "Decoder ID not registered in id_context_map_"
  );

  cache_[key].push_back(std::move(decoder));
  context_cache_[key].push_back(
    std::make_pair(
      VideoDecodeContext{
        videoFormat->min_num_decode_surfaces,
        videoFormat->coded_width,
        videoFormat->coded_height
      },
      it->second
    )
  );

  return true;
}


} // namespace facebook::torchcodec
