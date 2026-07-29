// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include "StableABICompat.h"

namespace facebook::torchcodec {

// Abstract io handler to read and write stuff. Implementations provide raw
// read/write/seek/get_size primitives over some backing store (an in-memory
// tensor, a file on disk, a Python file-like object, ...).
//
// Two kinds of consumers use it:
//   - FFmpeg code wraps an IOInterface in an AVIOContextHolder, which adapts it
//     into an FFmpeg AVIOContext.
//   - Non-FFmpeg code (the image encoders, WavDecoder) calls the primitives
//     directly.
class FORCE_PUBLIC_VISIBILITY IOInterface {
 public:
  virtual ~IOInterface() = default;

  virtual int read(uint8_t* /*buf*/, int /*size*/) {
    STD_TORCH_CHECK(false, "read() is not supported by this IOInterface");
  }

  virtual int write(const uint8_t* /*buf*/, int /*size*/) {
    STD_TORCH_CHECK(false, "write() is not supported by this IOInterface");
  }

  virtual int64_t seek(int64_t /*offset*/, int /*whence*/) {
    STD_TORCH_CHECK(false, "seek() is not supported by this IOInterface");
  }

  virtual int64_t get_size() {
    STD_TORCH_CHECK(false, "get_size() is not supported by this IOInterface");
  }
};

} // namespace facebook::torchcodec
