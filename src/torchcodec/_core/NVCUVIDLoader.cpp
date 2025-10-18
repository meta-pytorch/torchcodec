// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/NVCUVIDLoader.h"

#include <cstdio>

namespace facebook::torchcodec {

namespace {

#if defined(_WIN32)
constexpr const wchar_t* kLibName = L"nvcuvid.dll";
#else
constexpr const char* kLibName = "libnvcuvid.so";
#endif

template <typename T>
inline bool ResolveSymbol(NVCUVIDLoader::LibHandle handle, const char* name, T*& out) {
#if defined(_WIN32)
  FARPROC p = GetProcAddress(handle, name);
  out = reinterpret_cast<T*>(p);
#else
  void* p = dlsym(handle, name);
  out = reinterpret_cast<T*>(p);
#endif
  return out != nullptr;
}

} // namespace

NVCUVIDLoader& NVCUVIDLoader::instance() {
  static NVCUVIDLoader loader;
  return loader;
}

NVCUVIDLoader::~NVCUVIDLoader() {
#if defined(_WIN32)
  if (handle_) {
    FreeLibrary(handle_);
  }
#else
  if (handle_) {
    dlclose(handle_);
  }
#endif
}

bool NVCUVIDLoader::ensureLoaded() {
  if (loaded_) {
    return true;
  }
  if (!loadLibrary()) {
    return false;
  }
  loaded_ = resolveSymbols();
  return loaded_;
}

const NVCUVIDLoader::API& NVCUVIDLoader::api() {
  if (!ensureLoaded()) {
    // Keep the error message concise; callers should convert this to a
    // TORCH_CHECK with more context.
    std::fputs("Failed to load libnvcuvid and resolve required symbols\n", stderr);
  }
  return api_;
}

bool NVCUVIDLoader::loadLibrary() {
#if defined(_WIN32)
  handle_ = LoadLibraryW(kLibName);
#else
  handle_ = dlopen(kLibName, RTLD_NOW);
  if (!handle_) {
    // Fallback to common soname with version suffix used on some systems, as done by dali
    // https://github.com/NVIDIA/DALI/blob/a10cef187c0a5f27b6415df5d023c8057b9b43e2/dali/operators/video/dynlink_nvcuvid/dynlink_nvcuvid.cc#L35C18-L35C34
    handle_ = dlopen("libnvcuvid.so.1", RTLD_NOW);
  }
#endif
  return handle_ != nullptr;
}

bool NVCUVIDLoader::resolveSymbols() {
  bool ok = true;

  ok &= ResolveSymbol(handle_, "cuvidCreateVideoParser", api_.cuvidCreateVideoParser);
  ok &= ResolveSymbol(handle_, "cuvidParseVideoData", api_.cuvidParseVideoData);
  ok &= ResolveSymbol(handle_, "cuvidDestroyVideoParser", api_.cuvidDestroyVideoParser);

  ok &= ResolveSymbol(handle_, "cuvidGetDecoderCaps", api_.cuvidGetDecoderCaps);
  ok &= ResolveSymbol(handle_, "cuvidCreateDecoder", api_.cuvidCreateDecoder);
  ok &= ResolveSymbol(handle_, "cuvidDestroyDecoder", api_.cuvidDestroyDecoder);
  ok &= ResolveSymbol(handle_, "cuvidDecodePicture", api_.cuvidDecodePicture);

  ok &= ResolveSymbol(handle_, "cuvidMapVideoFrame", api_.cuvidMapVideoFrame);
  ok &= ResolveSymbol(handle_, "cuvidUnmapVideoFrame", api_.cuvidUnmapVideoFrame);

  return ok;
}

} // namespace facebook::torchcodec
