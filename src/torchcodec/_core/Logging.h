// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace facebook::torchcodec {

void setLogLevel(int level);
int getLogLevel();

namespace internal {
void log(const char* file, int line, const char* fmt, ...)
#ifndef _WIN32
    __attribute__((format(printf, 3, 4)))
#endif
    ;
} // namespace internal
} // namespace facebook::torchcodec

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define TC_LOG(...)                                                           \
  do {                                                                        \
    if (::facebook::torchcodec::getLogLevel() > 0) {                          \
      ::facebook::torchcodec::internal::log(__FILE__, __LINE__, __VA_ARGS__); \
    }                                                                         \
  } while (0)
