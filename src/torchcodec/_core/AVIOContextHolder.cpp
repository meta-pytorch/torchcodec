// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "AVIOContextHolder.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

// --------------------------------------------------------------------------
// FFmpeg AVIO callbacks — delegate to the wrapped IOInterface via opaque
// --------------------------------------------------------------------------

int AVIOContextHolder::read_callback(void* opaque, uint8_t* buf, int buf_size) {
  auto io = static_cast<IOInterface*>(opaque);
  int result = io->read(buf, buf_size);
  return result < 0 ? AVERROR_EOF : result;
}

int AVIOContextHolder::write_callback(
    void* opaque,
    const uint8_t* buf,
    int buf_size) {
  auto io = static_cast<IOInterface*>(opaque);
  return io->write(buf, buf_size);
}

int64_t
AVIOContextHolder::seek_callback(void* opaque, int64_t offset, int whence) {
  auto io = static_cast<IOInterface*>(opaque);
  if (whence == AVSEEK_SIZE) {
    int64_t size = io->get_size();
    // INT64_MAX means "unknown size" (e.g. streaming file-like objects).
    // Tell FFmpeg the size is unavailable rather than passing a bogus value.
    return size == INT64_MAX ? AVERROR(EIO) : size;
  }
  return io->seek(offset, whence);
}

// --------------------------------------------------------------------------
// AVIO context creation and lifecycle
// --------------------------------------------------------------------------

AVIOContextHolder::AVIOContextHolder(
    std::unique_ptr<IOInterface> io,
    bool is_for_writing,
    int buffer_size)
    : io_(std::move(io)) {
  STD_TORCH_CHECK(io_ != nullptr, "IOInterface must not be null");
  STD_TORCH_CHECK(
      buffer_size > 0,
      "Buffer size must be greater than 0; is " + std::to_string(buffer_size));
  auto buffer = static_cast<uint8_t*>(av_malloc(buffer_size));
  STD_TORCH_CHECK(
      buffer != nullptr,
      "Failed to allocate buffer of size " + std::to_string(buffer_size));

  avio_context_.reset(avio_alloc_context(
      buffer,
      buffer_size,
      /*write_flag=*/is_for_writing,
      /*opaque=*/io_.get(),
      is_for_writing ? nullptr : &read_callback,
      is_for_writing ? &write_callback : nullptr,
      &seek_callback));

  if (!avio_context_) {
    av_freep(&buffer);
    STD_TORCH_CHECK(false, "Failed to allocate AVIOContext");
  }
}

AVIOContextHolder::~AVIOContextHolder() {
  if (avio_context_) {
    av_freep(&avio_context_->buffer);
  }
}

AVIOContext* AVIOContextHolder::get_avio_context() {
  return avio_context_.get();
}

} // namespace facebook::torchcodec
