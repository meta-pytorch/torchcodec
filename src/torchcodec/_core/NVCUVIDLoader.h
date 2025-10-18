// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"

namespace facebook::torchcodec {

// Thin runtime loader for NVCUVID (NVDEC) symbols so we don't need to
// hard-link against libnvcuvid. This follows NVIDIA's guidance for dynamic
// loading.
class NVCUVIDLoader {
 public:
  struct API {
    // Parser
    CUresult(CUDAAPI* cuvidCreateVideoParser)(
        CUvideoparser*, CUVIDPARSERPARAMS*);
    CUresult(CUDAAPI* cuvidParseVideoData)(
        CUvideoparser, CUVIDSOURCEDATAPACKET*);
    CUresult(CUDAAPI* cuvidDestroyVideoParser)(CUvideoparser);

    // Decoder
    CUresult(CUDAAPI* cuvidGetDecoderCaps)(CUVIDDECODECAPS*);
    CUresult(CUDAAPI* cuvidCreateDecoder)(
        CUvideodecoder*, CUVIDDECODECREATEINFO*);
    CUresult(CUDAAPI* cuvidDestroyDecoder)(CUvideodecoder);
    CUresult(CUDAAPI* cuvidDecodePicture)(
        CUvideodecoder, CUVIDPICPARAMS*);

    // Frame mapping
    CUresult(CUDAAPI* cuvidMapVideoFrame)(
        CUvideodecoder,
        int,
        CUdeviceptr*,
        unsigned int*,
        CUVIDPROCPARAMS*);
    CUresult(CUDAAPI* cuvidUnmapVideoFrame)(
        CUvideodecoder, unsigned int /* DevPtr */);
  };

  // Singleton
  static NVCUVIDLoader& instance();

  // Returns true if the library is loaded and required symbols resolved.
  bool ensureLoaded();

  // Access resolved API. ensureLoaded() will be called implicitly; returns a
  // reference to a fully populated API or aborts if unavailable.
  const API& api();

 private:
  NVCUVIDLoader() = default;
  ~NVCUVIDLoader();
  NVCUVIDLoader(const NVCUVIDLoader&) = delete;
  NVCUVIDLoader& operator=(const NVCUVIDLoader&) = delete;

#if defined(_WIN32)
  using LibHandle = HMODULE;
#else
  using LibHandle = void*;
#endif

  LibHandle handle_ = nullptr;
  bool loaded_ = false;
  API api_{};

  bool loadLibrary();
  bool resolveSymbols();
};

} // namespace facebook::torchcodec
