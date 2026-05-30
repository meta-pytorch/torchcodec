// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <fstream>
#include <string>
#include "AVIOContextHolder.h"

namespace facebook::torchcodec {

// For reading from a file on disk. Unlike the other AVIOContextHolder
// subclasses, this one does NOT create an FFmpeg AVIOContext — it only
// provides the read/seek/getSize primitives for consumers like
// WavDecoder that do their own parsing.
class AVIOFileContext : public AVIOContextHolder {
 public:
  explicit AVIOFileContext(const std::string& path);

  int read(uint8_t* buf, int size) override;
  int64_t seek(int64_t offset, int whence) override;
  int64_t getSize() override;

 private:
  std::ifstream file_;
  int64_t fileSize_;
};

} // namespace facebook::torchcodec
