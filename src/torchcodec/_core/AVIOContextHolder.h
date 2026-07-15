// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

// The AVIOContextHolder is a base class for I/O backends. It serves as:
//
//   1. A generic I/O interface: derived classes override virtual methods
//      (read, write, seek, getSize) to implement their specific I/O.
//      These can be called directly by consumers like WavDecoder.
//
//   2. An FFmpeg AVIO adapter: calling createAVIOContext() sets up an
//      FFmpeg AVIOContext whose callbacks automatically delegate to the
//      virtual methods. This is used by SingleStreamDecoder and Encoder.
//
//   3. A smart pointer for the AVIOContext, freeing it and its buffer
//      on destruction.
class FORCE_PUBLIC_VISIBILITY AVIOContextHolder {
 public:
  virtual ~AVIOContextHolder();
  AVIOContext* get_avio_context();

  virtual int read(uint8_t* buf, int size);
  virtual int write(const uint8_t* buf, int size);
  virtual int64_t seek(int64_t offset, int whence);
  virtual int64_t get_size();

 protected:
  AVIOContextHolder() = default;

  // Sets up an FFmpeg AVIOContext whose callbacks delegate to the
  // virtual methods above. Derived classes that need FFmpeg AVIO
  // should call this in their constructor.
  void create_avio_context(
      bool is_for_writing,
      int buffer_size = default_buffer_size);

 private:
  static int read_callback(void* opaque, uint8_t* buf, int buf_size);
  static int write_callback(void* opaque, const uint8_t* buf, int buf_size);
  static int64_t seek_callback(void* opaque, int64_t offset, int whence);

  UniqueAVIOContext avio_context_;

  // Defaults to 64 KB
  static const int default_buffer_size = 64 * 1024;
};

} // namespace facebook::torchcodec
