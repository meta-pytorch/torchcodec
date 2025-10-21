// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/types.h>
#include <mutex>

#include "src/torchcodec/_core/CUDACommon.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/NVDECCache.h"

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

UniqueCUvideodecoder NVDECCache::getDecoder(CUVIDEOFORMAT* videoFormat) {
  CacheKey key(videoFormat);
  std::lock_guard<std::mutex> lock(cacheLock_);

  auto it = cache_.find(key);
  if (it != cache_.end()) {
    auto decoder = std::move(it->second);
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

  if (cache_.size() >= MAX_CACHE_SIZE) {
    return false;
  }

  cache_[key] = std::move(decoder);
  return true;
}

} // namespace facebook::torchcodec
