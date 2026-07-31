// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <fstream>
#include <string>
#include "IOInterface.h"

namespace facebook::torchcodec {

// file I/O on disk, backed by std::fstream. Opened for either
// reading or writing. Reading is used by consumers that parse bytes themselves
// (e.g. WavDecoder); writing is used by the image encoders.
class FileIO : public IOInterface {
 public:
  enum class Mode { Read, Write };
  FileIO(const std::string& path, Mode mode);

  int read(uint8_t* buf, int size) override;
  int write(const uint8_t* buf, int size) override;
  int64_t seek(int64_t offset, int whence) override;
  int64_t get_size() override;

 private:
  std::fstream file_;
  Mode mode_;
  int64_t file_size_ = 0;
};

} // namespace facebook::torchcodec
