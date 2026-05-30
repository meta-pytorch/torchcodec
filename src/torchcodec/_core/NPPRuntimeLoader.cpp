// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifdef FBCODE_CAFFE2 // See NVCUVIDRuntimeLoader.cpp.
#include "StableABICompat.h"

namespace facebook::torchcodec {
bool loadNPPLibrary() {
  return true;
}
} // namespace facebook::torchcodec
#else

#include "NPPRuntimeLoader.h"
#include "StableABICompat.h"

#include <npp.h>

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

// This file follows the same pattern as NVCUVIDRuntimeLoader.cpp.
// See the detailed explanation there.
//
// We dynamically load libnppicc (NPP image color conversion) at runtime
// to avoid a hard link-time dependency. This way, `import torchcodec`
// won't fail on machines without NPP installed.

/* clang-format off */
// Function pointer types
typedef NppStatus t_nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst[2], int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
/* clang-format on */

// Global function pointers - will be dynamically loaded
static t_nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx*
    dl_nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx = nullptr;

static tHandle g_nppicc_handle = nullptr;
static std::mutex g_npp_mutex;

static bool isLoaded() {
  return (g_nppicc_handle && dl_nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx);
}

template <typename T>
static T* bindFunction(tHandle handle, const char* functionName) {
#if defined(WIN64) || defined(_WIN64)
  return reinterpret_cast<T*>(GetProcAddress(handle, functionName));
#else
  return reinterpret_cast<T*>(dlsym(handle, functionName));
#endif
}

static bool _loadLibrary() {
#if defined(WIN64) || defined(_WIN64)
#if CUDART_VERSION >= 13000
  g_nppicc_handle = LoadLibraryA("nppicc64_13.dll");
#else
  g_nppicc_handle = LoadLibraryA("nppicc64_12.dll");
#endif
#else
  g_nppicc_handle = dlopen("libnppicc.so", RTLD_NOW);
  if (g_nppicc_handle == nullptr) {
#if CUDART_VERSION >= 13000
    g_nppicc_handle = dlopen("libnppicc.so.13", RTLD_NOW);
#else
    g_nppicc_handle = dlopen("libnppicc.so.12", RTLD_NOW);
#endif
  }
#endif
  if (g_nppicc_handle == nullptr) {
    return false;
  }

  return true;
}

bool loadNPPLibrary() {
  std::lock_guard<std::mutex> lock(g_npp_mutex);

  if (isLoaded()) {
    return true;
  }

  if (!_loadLibrary()) {
    return false;
  }

  dl_nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx =
      bindFunction<t_nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx>(
          g_nppicc_handle, "nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx");

  return isLoaded();
}

} // namespace facebook::torchcodec

extern "C" {

/* clang-format off */
NppStatus nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(
    const Npp8u* pSrc,
    int nSrcStep,
    Npp8u* pDst[2],
    int aDstStep[2],
    NppiSize oSizeROI,
    const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx,
      "nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx called but NPP not loaded!");
  return facebook::torchcodec::dl_nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(
      pSrc, nSrcStep, pDst, aDstStep, oSizeROI, aTwist, nppStreamCtx);
}

/* clang-format on */

} // extern "C"

#endif // FBCODE_CAFFE2
