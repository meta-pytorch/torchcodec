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
  AVIOContext* getAVIOContext();

  virtual int read(uint8_t* buf, int size);
  virtual int write(const uint8_t* buf, int size);
  virtual int64_t seek(int64_t offset, int whence);
  virtual int64_t getSize();

 protected:
  AVIOContextHolder() = default;

  // Sets up an FFmpeg AVIOContext whose callbacks delegate to the
  // virtual methods above. Derived classes that need FFmpeg AVIO
  // should call this in their constructor.
  void createAVIOContext(bool isForWriting, int bufferSize = defaultBufferSize);

 private:
  static int readCallback(void* opaque, uint8_t* buf, int buf_size);
  static int writeCallback(void* opaque, const uint8_t* buf, int buf_size);
  static int64_t seekCallback(void* opaque, int64_t offset, int whence);

  UniqueAVIOContext avioContext_;

  // Defaults to 64 KB
  static const int defaultBufferSize = 64 * 1024;
};

} // namespace facebook::torchcodec
