// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "FileIO.h"

#include <filesystem>
#include "StableABICompat.h"

namespace facebook::torchcodec {

FileIO::FileIO(const std::string& path, Mode mode) : mode_(mode) {
  if (mode_ == Mode::Read) {
    file_.open(path, std::ios::in | std::ios::binary);
    STD_TORCH_CHECK(file_.is_open(), "Failed to open file for reading: ", path);
    try {
      file_size_ = static_cast<int64_t>(std::filesystem::file_size(path));
    } catch (const std::filesystem::filesystem_error& e) {
      STD_TORCH_CHECK(
          false, "Failed to get file size for: ", path, ". Error: ", e.what());
    }
  } else {
    file_.open(path, std::ios::out | std::ios::binary | std::ios::trunc);
    STD_TORCH_CHECK(file_.is_open(), "Failed to open file for writing: ", path);
  }
}

int FileIO::read(uint8_t* buf, int size) {
  STD_TORCH_CHECK(mode_ == Mode::Read, "FileIO was not opened for reading");
  file_.read(reinterpret_cast<char*>(buf), size);
  auto bytes_read = static_cast<int>(file_.gcount());
  if (bytes_read == 0) {
    return -1;
  }
  return bytes_read;
}

int FileIO::write(const uint8_t* buf, int size) {
  STD_TORCH_CHECK(mode_ == Mode::Write, "FileIO was not opened for writing");
  file_.write(reinterpret_cast<const char*>(buf), size);
  STD_TORCH_CHECK(!file_.fail(), "Failed to write to file");
  return size;
}

int64_t FileIO::seek(int64_t offset, int whence) {
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
  if (mode_ == Mode::Read) {
    file_.seekg(offset, dir);
    STD_TORCH_CHECK(!file_.fail(), "Failed to seek in file");
    return static_cast<int64_t>(file_.tellg());
  }
  file_.seekp(offset, dir);
  STD_TORCH_CHECK(!file_.fail(), "Failed to seek in file");
  return static_cast<int64_t>(file_.tellp());
}

int64_t FileIO::get_size() {
  STD_TORCH_CHECK(
      mode_ == Mode::Read, "get_size() is only supported when reading");
  return file_size_;
}

} // namespace facebook::torchcodec
