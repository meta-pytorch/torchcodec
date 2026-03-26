// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Logging.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace facebook::torchcodec {
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
