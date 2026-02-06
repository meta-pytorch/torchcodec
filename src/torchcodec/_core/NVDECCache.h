// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <list>
#include <tuple>

#include <cuda.h>
#include <torch/types.h>

#include "NVCUVIDRuntimeLoader.h"
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

enum NVDECCacheType {
    Create,
    Reconfig,
    Reuse
};

using CUvideodecoderCache = 
  std::tuple<NVDECCacheType, UniqueCUvideodecoder, uint32_t>;

// A per-device cache for NVDEC decoders. There is one instance of this class
// per GPU device, and it is accessed through the static getCache() method.
class NVDECCache {
 public:
  static NVDECCache& getCache(const torch::Device& device);

  // Simple ID allocator for decoders - not strictly necessary, but useful for
  // reconfiguring decoders.
  uint32_t allocDecoderId();

  // Register a decoder ID with its maximum width and height.
  bool registerDecoderId(
    uint32_t decoderId,
    uint32_t ulMaxWidth,
    uint32_t ulMaxHeight,
    uint8_t  ulMaxNumDecodeSurfaces
  );

  // Get decoder from cache - returns nullptr if none available
  CUvideodecoderCache getDecoder(CUVIDEOFORMAT* videoFormat);

  // Return decoder to cache - returns true if added to cache
  bool returnDecoder(CUVIDEOFORMAT* videoFormat, UniqueCUvideodecoder decoder, uint32_t decoderId);

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

      explicit CacheKey(CUVIDEOFORMAT* videoFormat)
          : codecType(videoFormat->codec),
            width(videoFormat->coded_width),
            height(videoFormat->coded_height),
            chromaFormat(videoFormat->chroma_format),
            bitDepthLumaMinus8(videoFormat->bit_depth_luma_minus8),
            numDecodeSurfaces(videoFormat->min_num_decode_surfaces) {}

      CacheKey(const CacheKey&) = default;
      CacheKey& operator=(const CacheKey&) = default;

      bool operator<(const CacheKey& other) const {
          return std::tie(
            codecType, 
            chromaFormat, 
            bitDepthLumaMinus8
          ) <
          std::tie(
            other.codecType, 
            other.chromaFormat, 
            other.bitDepthLumaMinus8
          );
      }
  };

  struct VideoDecodeContext {
      uint8_t numDecodeSurfaces;
      uint32_t coded_width;
      uint32_t coded_height;
  };

  struct DecoderMaxWHContext {
      uint32_t decoderID;
      uint32_t ulMaxWidth;
      uint32_t ulMaxHeight;
      uint8_t  ulMaxNumDecodeSurfaces;
  };

  NVDECCache() = default;
  ~NVDECCache() = default;

  std::map<CacheKey, std::list<UniqueCUvideodecoder>> cache_;
  std::map<CacheKey, std::list<std::pair<VideoDecodeContext, DecoderMaxWHContext>>> context_cache_;
  std::map<uint32_t, DecoderMaxWHContext> id_context_map_;
  std::mutex cacheLock_;

  // Max number of cached decoders, per device
  static constexpr int MAX_CACHE_SIZE = 20;

  uint32_t nextId = 0;
};

} // namespace facebook::torchcodec
