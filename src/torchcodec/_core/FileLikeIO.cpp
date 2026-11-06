// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "FileLikeIO.h"

#include <algorithm>
#include <string>

#include "StableABICompat.h"

namespace facebook::torchcodec {

FileLikeIO::FileLikeIO(const FileLikeCallbacks& callbacks)
    : callbacks_(callbacks) {}

FileLikeIO::~FileLikeIO() {
  callbacks_.release(callbacks_.id_);
}

void FileLikeIO::check_callback_result(int64_t result, const char* method) {
  if (result != kFileLikeCallbackError) {
    return;
  }
  std::string message(1024, '\0');
  int64_t length = callbacks_.get_error(
      callbacks_.id_, message.data(), static_cast<int64_t>(message.size()));
  // get_error() never returns more than buf_len, but if it ever raised, ctypes
  // would hand us a garbage length.
  length = std::clamp<int64_t>(length, 0, message.size());
  message.resize(static_cast<size_t>(length));
  STD_TORCH_CHECK(
      false,
      "Calling the ",
      method,
      "() method of the file-like object raised an exception: ",
      message);
}

int FileLikeIO::read(uint8_t* buf, int size) {
  // The Python side fills buf unless it reaches EOF.
  int64_t num_bytes_read = callbacks_.read(callbacks_.id_, buf, size);
  check_callback_result(num_bytes_read, "read");
  return num_bytes_read == 0 ? -1 : static_cast<int>(num_bytes_read);
}

int FileLikeIO::write(const uint8_t* buf, int size) {
  int64_t result = callbacks_.write(callbacks_.id_, buf, size);
  check_callback_result(result, "write");
  return static_cast<int>(result);
}

int64_t FileLikeIO::seek(int64_t offset, int whence) {
  int64_t result = callbacks_.seek(callbacks_.id_, offset, whence);
  check_callback_result(result, "seek");
  return result;
}

int64_t FileLikeIO::get_size() {
  // Size of file-like is typically unknown, since the data is potentially
  // streaming.
  return INT64_MAX;
}

std::unique_ptr<IOInterface> adopt_file_like_context(
    int64_t file_like_context) {
  auto* callbacks = reinterpret_cast<const FileLikeCallbacks*>(
      static_cast<intptr_t>(file_like_context));
  STD_TORCH_CHECK(
      callbacks != nullptr, "file_like_context must be a valid pointer");
  return std::make_unique<FileLikeIO>(*callbacks);
}

} // namespace facebook::torchcodec
