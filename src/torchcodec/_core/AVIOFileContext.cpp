// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "AVIOFileContext.h"

#include <filesystem>
#include "StableABICompat.h"

namespace facebook::torchcodec {

AVIOFileContext::AVIOFileContext(const std::string& path)
    : file_(path, std::ios::binary) {
  STD_TORCH_CHECK(file_.is_open(), "Failed to open file: ", path);
  try {
    file_size_ = static_cast<int64_t>(std::filesystem::file_size(path));
  } catch (const std::filesystem::filesystem_error& e) {
    STD_TORCH_CHECK(
        false, "Failed to get file size for: ", path, ". Error: ", e.what());
  }
}

int AVIOFileContext::read(uint8_t* buf, int size) {
  file_.read(reinterpret_cast<char*>(buf), size);
  auto bytes_read = static_cast<int>(file_.gcount());
  if (bytes_read == 0) {
    return -1;
  }
  return bytes_read;
}

int64_t AVIOFileContext::seek(int64_t offset, int whence) {
  std::ios_base::seekdir dir;
  switch (whence) {
    case SEEK_SET:
      dir = std::ios::beg;
      break;
    case SEEK_CUR:
      dir = std::ios::cur;
      break;
    case SEEK_END:
      dir = std::ios::end;
      break;
    default:
      return -1;
  }
  file_.seekg(offset, dir);
  STD_TORCH_CHECK(!file_.fail(), "Failed to seek in file");
  return static_cast<int64_t>(file_.tellg());
}

int64_t AVIOFileContext::get_size() {
  return file_size_;
}

} // namespace facebook::torchcodec
