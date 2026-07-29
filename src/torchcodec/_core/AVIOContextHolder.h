// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include "FFMPEGCommon.h"
#include "IOInterface.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

// Adapts an FFmpeg-free IOInterface into an FFmpeg AVIOContext. It owns both
// the wrapped IOInterface and the AVIOContext (and the AVIOContext's buffer),
// freeing them on destruction. Used by SingleStreamDecoder and Encoder to
// feed/drain FFmpeg through a custom I/O backend (an in-memory tensor, a Python
// file-like object, ...): the AVIO callbacks delegate to the wrapped
// IOInterface's read/write/seek/get_size.
class FORCE_PUBLIC_VISIBILITY AVIOContextHolder {
 public:
  AVIOContextHolder(
      std::unique_ptr<IOInterface> io,
      bool is_for_writing,
      int buffer_size = default_buffer_size);
  ~AVIOContextHolder();

  AVIOContext* get_avio_context();

 private:
  static int read_callback(void* opaque, uint8_t* buf, int buf_size);
  static int write_callback(void* opaque, const uint8_t* buf, int buf_size);
  static int64_t seek_callback(void* opaque, int64_t offset, int whence);

  std::unique_ptr<IOInterface> io_;
  UniqueAVIOContext avio_context_;

  // Defaults to 64 KB
  static const int default_buffer_size = 64 * 1024;
};

} // namespace facebook::torchcodec
