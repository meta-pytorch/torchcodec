// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/NVCUVIDDynamicLoader.h"

// Include NVCUVID headers to get types
#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"

#include <dlfcn.h>
#include <torch/types.h>
#include <cstdio>
#include <mutex>

namespace facebook::torchcodec {

// Function typedefs
typedef CUresult CUDAAPI
tcuvidCreateVideoParser(CUvideoparser* pObj, CUVIDPARSERPARAMS* pParams);
typedef CUresult CUDAAPI
tcuvidParseVideoData(CUvideoparser obj, CUVIDSOURCEDATAPACKET* pPacket);
typedef CUresult CUDAAPI tcuvidDestroyVideoParser(CUvideoparser obj);
typedef CUresult CUDAAPI tcuvidGetDecoderCaps(CUVIDDECODECAPS* pdc);
typedef CUresult CUDAAPI
tcuvidCreateDecoder(CUvideodecoder* phDecoder, CUVIDDECODECREATEINFO* pdci);
typedef CUresult CUDAAPI tcuvidDestroyDecoder(CUvideodecoder hDecoder);
typedef CUresult CUDAAPI
tcuvidDecodePicture(CUvideodecoder hDecoder, CUVIDPICPARAMS* pPicParams);
typedef CUresult CUDAAPI tcuvidMapVideoFrame64(
    CUvideodecoder hDecoder,
    int nPicIdx,
    unsigned long long* pDevPtr,
    unsigned int* pPitch,
    CUVIDPROCPARAMS* pVPP);
typedef CUresult CUDAAPI
tcuvidUnmapVideoFrame64(CUvideodecoder hDecoder, unsigned long long DevPtr);

// Global function pointers
static tcuvidCreateVideoParser* _dlcuvidCreateVideoParser = nullptr;
static tcuvidParseVideoData* _dlcuvidParseVideoData = nullptr;
static tcuvidDestroyVideoParser* _dlcuvidDestroyVideoParser = nullptr;
static tcuvidGetDecoderCaps* _dlcuvidGetDecoderCaps = nullptr;
static tcuvidCreateDecoder* _dlcuvidCreateDecoder = nullptr;
static tcuvidDestroyDecoder* _dlcuvidDestroyDecoder = nullptr;
static tcuvidDecodePicture* _dlcuvidDecodePicture = nullptr;
static tcuvidMapVideoFrame64* _dlcuvidMapVideoFrame64 = nullptr;
static tcuvidUnmapVideoFrame64* _dlcuvidUnmapVideoFrame64 = nullptr;

static void* g_nvcuvid_handle = nullptr;
static std::mutex g_nvcuvid_mutex;

template <typename T>
T* loadFunction(const char* functionName) {
  return reinterpret_cast<T*>(dlsym(g_nvcuvid_handle, functionName));
}

bool initNVCUVID() {
  std::lock_guard<std::mutex> lock(g_nvcuvid_mutex);

  if (g_nvcuvid_handle != nullptr) {
    return true; // Already loaded
  }

  g_nvcuvid_handle = dlopen("libnvcuvid.so", RTLD_NOW);
  if (g_nvcuvid_handle == nullptr) {
    g_nvcuvid_handle = dlopen("libnvcuvid.so.1", RTLD_NOW);
    if (g_nvcuvid_handle == nullptr) {
      return false;
    }
  }

  // Load all function pointers
  _dlcuvidCreateVideoParser =
      loadFunction<tcuvidCreateVideoParser>("cuvidCreateVideoParser");
  _dlcuvidParseVideoData =
      loadFunction<tcuvidParseVideoData>("cuvidParseVideoData");
  _dlcuvidDestroyVideoParser =
      loadFunction<tcuvidDestroyVideoParser>("cuvidDestroyVideoParser");
  _dlcuvidGetDecoderCaps =
      loadFunction<tcuvidGetDecoderCaps>("cuvidGetDecoderCaps");
  _dlcuvidCreateDecoder =
      loadFunction<tcuvidCreateDecoder>("cuvidCreateDecoder");
  _dlcuvidDestroyDecoder =
      loadFunction<tcuvidDestroyDecoder>("cuvidDestroyDecoder");
  _dlcuvidDecodePicture =
      loadFunction<tcuvidDecodePicture>("cuvidDecodePicture");
  _dlcuvidMapVideoFrame64 =
      loadFunction<tcuvidMapVideoFrame64>("cuvidMapVideoFrame64");
  _dlcuvidUnmapVideoFrame64 =
      loadFunction<tcuvidUnmapVideoFrame64>("cuvidUnmapVideoFrame64");

  // Check if all critical functions loaded successfully
  return (_dlcuvidCreateVideoParser && _dlcuvidParseVideoData &&
          _dlcuvidDestroyVideoParser && _dlcuvidGetDecoderCaps &&
          _dlcuvidCreateDecoder && _dlcuvidDestroyDecoder &&
          _dlcuvidDecodePicture && _dlcuvidMapVideoFrame64 &&
          _dlcuvidUnmapVideoFrame64);
}

bool isNVCUVIDLoaded() {
  return g_nvcuvid_handle != nullptr && _dlcuvidCreateDecoder != nullptr;
}


} // namespace facebook::torchcodec

// Provide C-style wrapper functions that replace the original NVCUVID functions
extern "C" {

CUresult CUDAAPI
cuvidCreateVideoParser(CUvideoparser* pObj, CUVIDPARSERPARAMS* pParams) {
  TORCH_CHECK(
      facebook::torchcodec::_dlcuvidCreateVideoParser,
      "cuvidCreateVideoParser called but NVCUVID not loaded!");
  return facebook::torchcodec::_dlcuvidCreateVideoParser(pObj, pParams);
}

CUresult CUDAAPI
cuvidParseVideoData(CUvideoparser obj, CUVIDSOURCEDATAPACKET* pPacket) {
  TORCH_CHECK(
      facebook::torchcodec::_dlcuvidParseVideoData,
      "cuvidParseVideoData called but NVCUVID not loaded!");
  return facebook::torchcodec::_dlcuvidParseVideoData(obj, pPacket);
}

CUresult CUDAAPI cuvidDestroyVideoParser(CUvideoparser obj) {
  TORCH_CHECK(
      facebook::torchcodec::_dlcuvidDestroyVideoParser,
      "cuvidDestroyVideoParser called but NVCUVID not loaded!");
  return facebook::torchcodec::_dlcuvidDestroyVideoParser(obj);
}

CUresult CUDAAPI cuvidGetDecoderCaps(CUVIDDECODECAPS* pdc) {
  TORCH_CHECK(
      facebook::torchcodec::_dlcuvidGetDecoderCaps,
      "cuvidGetDecoderCaps called but NVCUVID not loaded!");
  return facebook::torchcodec::_dlcuvidGetDecoderCaps(pdc);
}

CUresult CUDAAPI
cuvidCreateDecoder(CUvideodecoder* phDecoder, CUVIDDECODECREATEINFO* pdci) {
  TORCH_CHECK(
      facebook::torchcodec::_dlcuvidCreateDecoder,
      "cuvidCreateDecoder called but NVCUVID not loaded!");
  return facebook::torchcodec::_dlcuvidCreateDecoder(phDecoder, pdci);
}

CUresult CUDAAPI cuvidDestroyDecoder(CUvideodecoder hDecoder) {
  TORCH_CHECK(
      facebook::torchcodec::_dlcuvidDestroyDecoder,
      "cuvidDestroyDecoder called but NVCUVID not loaded!");
  return facebook::torchcodec::_dlcuvidDestroyDecoder(hDecoder);
}

CUresult CUDAAPI
cuvidDecodePicture(CUvideodecoder hDecoder, CUVIDPICPARAMS* pPicParams) {
  TORCH_CHECK(
      facebook::torchcodec::_dlcuvidDecodePicture,
      "cuvidDecodePicture called but NVCUVID not loaded!");
  return facebook::torchcodec::_dlcuvidDecodePicture(hDecoder, pPicParams);
}

CUresult CUDAAPI cuvidMapVideoFrame64(
    CUvideodecoder hDecoder,
    int nPicIdx,
    unsigned long long* pDevPtr,
    unsigned int* pPitch,
    CUVIDPROCPARAMS* pVPP) {
  TORCH_CHECK(
      facebook::torchcodec::_dlcuvidMapVideoFrame64,
      "cuvidMapVideoFrame64 called but NVCUVID not loaded!");
  return facebook::torchcodec::_dlcuvidMapVideoFrame64(
      hDecoder, nPicIdx, pDevPtr, pPitch, pVPP);
}

CUresult CUDAAPI
cuvidUnmapVideoFrame64(CUvideodecoder hDecoder, unsigned long long DevPtr) {
  TORCH_CHECK(
      facebook::torchcodec::_dlcuvidUnmapVideoFrame64,
      "cuvidUnmapVideoFrame64 called but NVCUVID not loaded!");
  return facebook::torchcodec::_dlcuvidUnmapVideoFrame64(hDecoder, DevPtr);
}

} // extern "C"
