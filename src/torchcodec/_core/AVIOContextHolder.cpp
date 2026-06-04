// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "AVIOContextHolder.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

// --------------------------------------------------------------------------
// FFmpeg AVIO callbacks — delegate to virtual methods via opaque=this
// --------------------------------------------------------------------------

int AVIOContextHolder::readCallback(void* opaque, uint8_t* buf, int buf_size) {
  auto self = static_cast<AVIOContextHolder*>(opaque);
  int result = self->read(buf, buf_size);
  return result < 0 ? AVERROR_EOF : result;
}

int AVIOContextHolder::writeCallback(
    void* opaque,
    const uint8_t* buf,
    int buf_size) {
  auto self = static_cast<AVIOContextHolder*>(opaque);
  return self->write(buf, buf_size);
}

int64_t
AVIOContextHolder::seekCallback(void* opaque, int64_t offset, int whence) {
  auto self = static_cast<AVIOContextHolder*>(opaque);
  if (whence == AVSEEK_SIZE) {
    int64_t size = self->getSize();
    // INT64_MAX means "unknown size" (e.g. streaming file-like objects).
    // Tell FFmpeg the size is unavailable rather than passing a bogus value.
    return size == INT64_MAX ? AVERROR(EIO) : size;
  }
  return self->seek(offset, whence);
}

// --------------------------------------------------------------------------
// AVIO context creation and lifecycle
// --------------------------------------------------------------------------

void AVIOContextHolder::createAVIOContext(bool isForWriting, int bufferSize) {
  STD_TORCH_CHECK(
      bufferSize > 0,
      "Buffer size must be greater than 0; is " + std::to_string(bufferSize));
  auto buffer = static_cast<uint8_t*>(av_malloc(bufferSize));
  STD_TORCH_CHECK(
      buffer != nullptr,
      "Failed to allocate buffer of size " + std::to_string(bufferSize));

  avioContext_.reset(avioAllocContext(
      buffer,
      bufferSize,
      /*write_flag=*/isForWriting,
      /*opaque=*/this,
      isForWriting ? nullptr : &readCallback,
      isForWriting ? &writeCallback : nullptr,
      &seekCallback));

  if (!avioContext_) {
    av_freep(&buffer);
    STD_TORCH_CHECK(false, "Failed to allocate AVIOContext");
  }
}

AVIOContextHolder::~AVIOContextHolder() {
  if (avioContext_) {
    av_freep(&avioContext_->buffer);
  }
}

AVIOContext* AVIOContextHolder::getAVIOContext() {
  return avioContext_.get();
}

// --------------------------------------------------------------------------
// Default virtual method implementations
// --------------------------------------------------------------------------

int AVIOContextHolder::read(uint8_t*, int) {
  STD_TORCH_CHECK(false, "read() is not supported by this AVIOContextHolder");
}

int AVIOContextHolder::write(const uint8_t*, int) {
  STD_TORCH_CHECK(false, "write() is not supported by this AVIOContextHolder");
}

int64_t AVIOContextHolder::seek(int64_t, int) {
  STD_TORCH_CHECK(false, "seek() is not supported by this AVIOContextHolder");
}

int64_t AVIOContextHolder::getSize() {
  STD_TORCH_CHECK(
      false, "getSize() is not supported by this AVIOContextHolder");
}

} // namespace facebook::torchcodec
