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

CUvideodecoderCache NVDECCache::getDecoder(CUVIDEOFORMAT* videoFormat) {
  NVDECCacheType cache_type = NVDECCacheType::Create;
  CacheKey key(videoFormat);
  std::lock_guard<std::mutex> lock(cacheLock_);

  auto it = cache_.find(key);
  if (it != cache_.end()) {
    auto it2 = context_cache_.find(key);
    TORCH_CHECK(
      it2 != context_cache_.end(),
      "Decoder context cache inconsistency detected."
    );

    // We first check if the cached decoder can be reused as is. If the number of
    // surfaces allocated for the cached decoder is equal to the requested number of
    // surfaces, and the coded dimensions also match, then we can reuse it directly.
    // Otherwise, we need to reconfigure it.
    if (
      it2->second.numDecodeSurfaces == videoFormat->min_num_decode_surfaces &&
      it2->second.coded_width == videoFormat->coded_width &&
      it2->second.coded_height == videoFormat->coded_height
    ) {
      cache_type = NVDECCacheType::Reuse;
    }
    else {
      cache_type = NVDECCacheType::Reconfig;
    }
    
    auto decoder = std::move(it->second);
    cache_.erase(it);
    context_cache_.erase(it2);

    return std::make_pair(cache_type, std::move(decoder));
  }

  return std::make_pair(cache_type, nullptr);
}

bool NVDECCache::returnDecoder(
    CUVIDEOFORMAT* videoFormat,
    UniqueCUvideodecoder decoder) {
  if (!decoder) {
    return false;
  }

  CacheKey key(videoFormat);
  std::lock_guard<std::mutex> lock(cacheLock_);

  if (cache_.size() >= MAX_CACHE_SIZE) {
    return false;
  }

  cache_[key] = std::move(decoder);
  context_cache_[key].numDecodeSurfaces = videoFormat->min_num_decode_surfaces;
  context_cache_[key].coded_width = videoFormat->coded_width;
  context_cache_[key].coded_height = videoFormat->coded_height;
  return true;
}


} // namespace facebook::torchcodec
