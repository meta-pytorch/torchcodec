// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Logging.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <filesystem>

namespace facebook::torchcodec {

static std::atomic<int> gLogLevel{static_cast<int>(LogLevel::OFF)};

void setLogLevel(LogLevel level) {
  gLogLevel.store(static_cast<int>(level), std::memory_order_relaxed);
}

LogLevel getLogLevel() {
  return static_cast<LogLevel>(gLogLevel.load(std::memory_order_relaxed));
}

namespace internal {

void log(const char* file, int line, const char* fmt, ...) {
  auto basename = std::filesystem::path(file).filename().string();

  std::fprintf(stderr, "[torchcodec %s:%d] ", basename.c_str(), line);

  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);

  std::fprintf(stderr, "\n");
}

} // namespace internal
} // namespace facebook::torchcodec
