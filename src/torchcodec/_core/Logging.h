// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace facebook::torchcodec {

// Keep in sync with _LogLevel in torchcodec/_logging.py.
enum class LogLevel : int {
  OFF = 0,
  ALL = 1,
};

void setCppLogLevel(LogLevel level);

// getLogLevel doesn't have "Cpp" in the name because it's the source of
// truth for both Python and C++ log levels.
LogLevel getLogLevel();

namespace internal {
void log(const char* file, int line, const char* fmt, ...)
#ifndef _WIN32
    __attribute__((format(printf, 3, 4)))
#endif
    ;
} // namespace internal
} // namespace facebook::torchcodec

#define TC_LOG(...)                                                           \
  do {                                                                        \
    if (::facebook::torchcodec::getLogLevel() !=                              \
        ::facebook::torchcodec::LogLevel::OFF) {                              \
      ::facebook::torchcodec::internal::log(__FILE__, __LINE__, __VA_ARGS__); \
    }                                                                         \
  } while (0)
