// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// Torch-free error-checking macros and symbol-visibility macros for the core.
// These replace StableABICompat.h's STD_TORCH_CHECK / STABLE_CHECK_INDEX in the
// CPU core so it does not depend on torch headers. (The CUDA device interfaces
// and the custom-ops adapter still use StableABICompat.h / torch directly.)
//
//   TC_CHECK(cond, msg...)        -> throws std::runtime_error (Python
//                                    RuntimeError) with the streamed message.
//   TC_CHECK_INDEX(cond, msg...)  -> throws std::out_of_range (Python
//                                    IndexError).

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

// Symbol visibility for the shared library.
#ifdef _WIN32
#define FORCE_PUBLIC_VISIBILITY __declspec(dllexport)
#else
#define FORCE_PUBLIC_VISIBILITY __attribute__((visibility("default")))
#endif

// Flag for any API that third-party libraries may call; ensures the symbol is
// always public.
#ifdef _WIN32
#define TORCHCODEC_THIRD_PARTY_API
#else
#define TORCHCODEC_THIRD_PARTY_API __attribute__((visibility("default")))
#endif

namespace facebook::torchcodec::detail {

template <typename... Args>
std::string buildCheckMessage(Args&&... args) {
  std::ostringstream oss;
  // initializer-list expansion (safe and warning-free even with zero args).
  int dummy[] = {0, ((void)(oss << std::forward<Args>(args)), 0)...};
  (void)dummy;
  return oss.str();
}

template <typename... Args>
[[noreturn]] void throwCheckFailure(Args&&... args) {
  throw std::runtime_error(buildCheckMessage(std::forward<Args>(args)...));
}

template <typename... Args>
[[noreturn]] void throwIndexFailure(Args&&... args) {
  throw std::out_of_range(buildCheckMessage(std::forward<Args>(args)...));
}

} // namespace facebook::torchcodec::detail

#define TC_CHECK(cond, ...)                                            \
  do {                                                                 \
    if (!(cond)) {                                                     \
      ::facebook::torchcodec::detail::throwCheckFailure(__VA_ARGS__);  \
    }                                                                  \
  } while (false)

#define TC_CHECK_INDEX(cond, ...)                                      \
  do {                                                                 \
    if (!(cond)) {                                                     \
      ::facebook::torchcodec::detail::throwIndexFailure(__VA_ARGS__);  \
    }                                                                  \
  } while (false)
