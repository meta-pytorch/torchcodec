// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifdef FBCODE_CAFFE2
// No need to do anything on fbcode. NVCUVID is available there, we can take a
// hard dependency on it.
// The FBCODE_CAFFE2 macro is defined in the upstream fbcode build of torch, so
// we can rely on it, that's what torch does too.

namespace facebook::torchcodec {
bool loadNVCUVIDLibrary() {
  return true;
}
} // namespace facebook::torchcodec
#else

#include "src/torchcodec/_core/NVCUVIDRuntimeLoader.h"

#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"

#include <torch/types.h>
#include <cstdio>
#include <mutex>

#if defined(WIN64) || defined(_WIN64)
#include <windows.h>
typedef HMODULE tHandle;
#else
#include <dlfcn.h>
typedef void* tHandle;
#endif

namespace facebook::torchcodec {

/* clang-format off */
// This file defines the logic to load the NVCUVID library **at runtime**,
// along with the corresponding NVCUVID functions that we'll need.
//
// We do this because we *do not want* to link (statically or dynamically)
// against libnvcuvid.so: it is not always available on the users machine! If we
// were to link against libnvcuvid.so, that would mean that our
// libtorchcodec_coreN.so would try to look for it when loaded at import time.
// And if it's not on the users machine, that causes `import torchcodec` to
// fail. Source: that's what we did, and we got user reports.
//
// So, we don't link against libnvcuvid.so. But we still want to call its
// functions. So here's how it's done, we'll use cuvidCreateVideoParser as an
// example, but it works the same for all. We are largely following the
// instructions from the NVCUVID docs:
// https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html#dynamic-loading-nvidia-components
//
// This:
// typedef CUresult CUDAAPI tcuvidCreateVideoParser(CUvideoparser*, CUVIDPARSERPARAMS*);
// defines tcuvidCreateVideoParser, which is the *type* of a *function*.
// We define such a function of that type just below with:
// static tcuvidCreateVideoParser* dl_cuvidCreateVideoParser = nullptr;
// "dl" is for "dynamically loaded. For now dl_cuvidCreateVideoParser is
// nullptr, but later it will be a proper function [pointer] that can be called
// with dl_cuvidCreateVideoParser(...);
//
// For that to happen we need to call loadNVCUVIDLibrary(): in there, we first
// dlopen(libnvcuvid.so) which loads the .so somewhere in memory. Then we call
// dlsym(...), which binds dl_cuvidCreateVideoParser to its actual address: it
// literally sets the value of the dl_cuvidCreateVideoParser pointer to the
// address of the actual code section. If all went well, by now, we can safely
// call dl_cuvidCreateVideoParser(...);
// All of that happens at runtime *after* import time, when the first instance
// of the Beta CUDA interface is created, i.e. only when the user explicitly
// requests it.
//
// At the bottom of this file we have an `extern "C"` section with function
// definitions like:
//
// CUresult CUDAAPI cuvidCreateVideoParser(
//  CUvideoparser* videoParser,
//  CUVIDPARSERPARAMS* parserParams)  {...}
//
// These are the actual functions that are compiled against and called by the
// Beta CUDA interface code. Crucially, these functions signature match exactly
// the NVCUVID functions (as defined in cuviddec.h). Inside of
// cuvidCreateVideoParser(...) we simply call the dl_cuvidCreateVideoParser
// function [pointer] that we dynamically loaded earlier.
//
// At runtime, within the Beta CUDA interface code we have a fallback mechanism
// to switch back to the CPU backend if any of the NVCUVID functions are not
// available, or if libnvcuvid.so itself couldn't be found. This is what FFmpeg
// does too.


// Function pointers types
typedef CUresult CUDAAPI tcuvidCreateVideoParser(CUvideoparser*, CUVIDPARSERPARAMS*);
typedef CUresult CUDAAPI tcuvidParseVideoData(CUvideoparser, CUVIDSOURCEDATAPACKET*);
typedef CUresult CUDAAPI tcuvidDestroyVideoParser(CUvideoparser);
typedef CUresult CUDAAPI tcuvidGetDecoderCaps(CUVIDDECODECAPS*);
typedef CUresult CUDAAPI tcuvidCreateDecoder(CUvideodecoder*, CUVIDDECODECREATEINFO*);
typedef CUresult CUDAAPI tcuvidDestroyDecoder(CUvideodecoder);
typedef CUresult CUDAAPI tcuvidDecodePicture(CUvideodecoder, CUVIDPICPARAMS*);
typedef CUresult CUDAAPI tcuvidMapVideoFrame(CUvideodecoder, int, unsigned int*, unsigned int*, CUVIDPROCPARAMS*);
typedef CUresult CUDAAPI tcuvidUnmapVideoFrame(CUvideodecoder, unsigned int);
typedef CUresult CUDAAPI tcuvidMapVideoFrame64(CUvideodecoder, int, unsigned long long*, unsigned int*, CUVIDPROCPARAMS*);
typedef CUresult CUDAAPI tcuvidUnmapVideoFrame64(CUvideodecoder, unsigned long long);
/* clang-format on */

// Global function pointers - will be dynamically loaded
static tcuvidCreateVideoParser* dl_cuvidCreateVideoParser = nullptr;
static tcuvidParseVideoData* dl_cuvidParseVideoData = nullptr;
static tcuvidDestroyVideoParser* dl_cuvidDestroyVideoParser = nullptr;
static tcuvidGetDecoderCaps* dl_cuvidGetDecoderCaps = nullptr;
static tcuvidCreateDecoder* dl_cuvidCreateDecoder = nullptr;
static tcuvidDestroyDecoder* dl_cuvidDestroyDecoder = nullptr;
static tcuvidDecodePicture* dl_cuvidDecodePicture = nullptr;
static tcuvidMapVideoFrame* dl_cuvidMapVideoFrame = nullptr;
static tcuvidUnmapVideoFrame* dl_cuvidUnmapVideoFrame = nullptr;
static tcuvidMapVideoFrame64* dl_cuvidMapVideoFrame64 = nullptr;
static tcuvidUnmapVideoFrame64* dl_cuvidUnmapVideoFrame64 = nullptr;

static tHandle g_nvcuvid_handle = nullptr;
static std::mutex g_nvcuvid_mutex;

bool isLoaded() {
  return (
      g_nvcuvid_handle && dl_cuvidCreateVideoParser && dl_cuvidParseVideoData &&
      dl_cuvidDestroyVideoParser && dl_cuvidGetDecoderCaps &&
      dl_cuvidCreateDecoder && dl_cuvidDestroyDecoder &&
      dl_cuvidDecodePicture && dl_cuvidMapVideoFrame &&
      dl_cuvidUnmapVideoFrame && dl_cuvidMapVideoFrame64 &&
      dl_cuvidUnmapVideoFrame64);
}

template <typename T>
T* bindFunction(const char* functionName) {
#if defined(WIN64) || defined(_WIN64)
  return reinterpret_cast<T*>(GetProcAddress(g_nvcuvid_handle, functionName));
#else
  return reinterpret_cast<T*>(dlsym(g_nvcuvid_handle, functionName));
#endif
}

bool _loadLibrary() {
  // Helper that just calls dlopen or equivalent on Windows. In a separate
  // function because of the #ifdef uglyness.
#if defined(WIN64) || defined(_WIN64)
#ifdef UNICODE
  static LPCWSTR nvcuvidDll = L"nvcuvid.dll";
#else
  static LPCSTR nvcuvidDll = "nvcuvid.dll";
#endif
  g_nvcuvid_handle = LoadLibrary(nvcuvidDll);
  if (g_nvcuvid_handle == nullptr) {
    return false;
  }
#else
  g_nvcuvid_handle = dlopen("libnvcuvid.so", RTLD_NOW);
  if (g_nvcuvid_handle == nullptr) {
    g_nvcuvid_handle = dlopen("libnvcuvid.so.1", RTLD_NOW);
  }
  if (g_nvcuvid_handle == nullptr) {
    return false;
  }
#endif

  return true;
}

bool loadNVCUVIDLibrary() {
  // Loads NVCUVID library and all required function pointers.
  // Returns true on success, false on failure.
  std::lock_guard<std::mutex> lock(g_nvcuvid_mutex);

  if (isLoaded()) {
    return true;
  }

  if (!_loadLibrary()) {
    return false;
  }

  // Load all function pointers. They'll be set to nullptr if not found.
  dl_cuvidCreateVideoParser =
      bindFunction<tcuvidCreateVideoParser>("cuvidCreateVideoParser");
  dl_cuvidParseVideoData =
      bindFunction<tcuvidParseVideoData>("cuvidParseVideoData");
  dl_cuvidDestroyVideoParser =
      bindFunction<tcuvidDestroyVideoParser>("cuvidDestroyVideoParser");
  dl_cuvidGetDecoderCaps =
      bindFunction<tcuvidGetDecoderCaps>("cuvidGetDecoderCaps");
  dl_cuvidCreateDecoder =
      bindFunction<tcuvidCreateDecoder>("cuvidCreateDecoder");
  dl_cuvidDestroyDecoder =
      bindFunction<tcuvidDestroyDecoder>("cuvidDestroyDecoder");
  dl_cuvidDecodePicture =
      bindFunction<tcuvidDecodePicture>("cuvidDecodePicture");
  dl_cuvidMapVideoFrame =
      bindFunction<tcuvidMapVideoFrame>("cuvidMapVideoFrame");
  dl_cuvidUnmapVideoFrame =
      bindFunction<tcuvidUnmapVideoFrame>("cuvidUnmapVideoFrame");
  dl_cuvidMapVideoFrame64 =
      bindFunction<tcuvidMapVideoFrame64>("cuvidMapVideoFrame64");
  dl_cuvidUnmapVideoFrame64 =
      bindFunction<tcuvidUnmapVideoFrame64>("cuvidUnmapVideoFrame64");

  return isLoaded();
}

} // namespace facebook::torchcodec

extern "C" {

CUresult CUDAAPI cuvidCreateVideoParser(
    CUvideoparser* videoParser,
    CUVIDPARSERPARAMS* parserParams) {
  TORCH_CHECK(
      facebook::torchcodec::dl_cuvidCreateVideoParser,
      "cuvidCreateVideoParser called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidCreateVideoParser(
      videoParser, parserParams);
}

CUresult CUDAAPI cuvidParseVideoData(
    CUvideoparser videoParser,
    CUVIDSOURCEDATAPACKET* cuvidPacket) {
  TORCH_CHECK(
      facebook::torchcodec::dl_cuvidParseVideoData,
      "cuvidParseVideoData called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidParseVideoData(videoParser, cuvidPacket);
}

CUresult CUDAAPI cuvidDestroyVideoParser(CUvideoparser videoParser) {
  TORCH_CHECK(
      facebook::torchcodec::dl_cuvidDestroyVideoParser,
      "cuvidDestroyVideoParser called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidDestroyVideoParser(videoParser);
}

CUresult CUDAAPI cuvidGetDecoderCaps(CUVIDDECODECAPS* caps) {
  TORCH_CHECK(
      facebook::torchcodec::dl_cuvidGetDecoderCaps,
      "cuvidGetDecoderCaps called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidGetDecoderCaps(caps);
}

CUresult CUDAAPI cuvidCreateDecoder(
    CUvideodecoder* decoder,
    CUVIDDECODECREATEINFO* decoderParams) {
  TORCH_CHECK(
      facebook::torchcodec::dl_cuvidCreateDecoder,
      "cuvidCreateDecoder called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidCreateDecoder(decoder, decoderParams);
}

CUresult CUDAAPI cuvidDestroyDecoder(CUvideodecoder decoder) {
  TORCH_CHECK(
      facebook::torchcodec::dl_cuvidDestroyDecoder,
      "cuvidDestroyDecoder called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidDestroyDecoder(decoder);
}

CUresult CUDAAPI
cuvidDecodePicture(CUvideodecoder decoder, CUVIDPICPARAMS* picParams) {
  TORCH_CHECK(
      facebook::torchcodec::dl_cuvidDecodePicture,
      "cuvidDecodePicture called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidDecodePicture(decoder, picParams);
}

#if !defined(__CUVID_DEVPTR64) || defined(__CUVID_INTERNAL)
// We need to protect the definition of the 32bit versions under the above
// conditions (see cuviddec.h). Defining them unconditionally would cause
// conflict compilation errors when cuviddec.h redefines those to the 64bit
// versions.
CUresult CUDAAPI cuvidMapVideoFrame(
    CUvideodecoder decoder,
    int pixIndex,
    unsigned int* framePtr,
    unsigned int* pitch,
    CUVIDPROCPARAMS* procParams) {
  TORCH_CHECK(
      facebook::torchcodec::dl_cuvidMapVideoFrame,
      "cuvidMapVideoFrame called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidMapVideoFrame(
      decoder, pixIndex, framePtr, pitch, procParams);
}

CUresult CUDAAPI
cuvidUnmapVideoFrame(CUvideodecoder decoder, unsigned int framePtr) {
  TORCH_CHECK(
      facebook::torchcodec::dl_cuvidUnmapVideoFrame,
      "cuvidUnmapVideoFrame called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidUnmapVideoFrame(decoder, framePtr);
}
#endif

CUresult CUDAAPI cuvidMapVideoFrame64(
    CUvideodecoder decoder,
    int pixIndex,
    unsigned long long* framePtr,
    unsigned int* pitch,
    CUVIDPROCPARAMS* procParams) {
  TORCH_CHECK(
      facebook::torchcodec::dl_cuvidMapVideoFrame64,
      "cuvidMapVideoFrame64 called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidMapVideoFrame64(
      decoder, pixIndex, framePtr, pitch, procParams);
}

CUresult CUDAAPI
cuvidUnmapVideoFrame64(CUvideodecoder decoder, unsigned long long framePtr) {
  TORCH_CHECK(
      facebook::torchcodec::dl_cuvidUnmapVideoFrame64,
      "cuvidUnmapVideoFrame64 called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidUnmapVideoFrame64(decoder, framePtr);
}

} // extern "C"

#endif // FBCODE_CAFFE2
