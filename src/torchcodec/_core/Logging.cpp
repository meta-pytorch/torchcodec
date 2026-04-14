// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Logging.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace facebook::torchcodec {

// Log level: 0 = OFF (default), >0 = enabled.
// Currently only OFF and ALL (1) are used; more granular levels can be added
// later without changing this interface.
static std::atomic<int> gLogLevel{0};

void setLogLevel(int level) {
  gLogLevel.store(level, std::memory_order_relaxed);
}

int getLogLevel() {
  return gLogLevel.load(std::memory_order_relaxed);
}

namespace internal {

void log(const char* file, int line, const char* fmt, ...) {
  // Strip path prefix to show only the filename.
  const char* basename = std::strrchr(file, '/');
  if (basename != nullptr) {
    basename += 1;
  } else {
    basename = file;
  }

  std::fprintf(stderr, "[torchcodec %s:%d] ", basename, line);

  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);

  std::fprintf(stderr, "\n");
}

} // namespace internal
} // namespace facebook::torchcodec
