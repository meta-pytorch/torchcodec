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

static std::atomic<int> g_log_level{static_cast<int>(LogLevel::OFF)};

void set_cpp_log_level(LogLevel level) {
  g_log_level.store(static_cast<int>(level), std::memory_order_relaxed);
}

LogLevel get_log_level() {
  return static_cast<LogLevel>(g_log_level.load(std::memory_order_relaxed));
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
