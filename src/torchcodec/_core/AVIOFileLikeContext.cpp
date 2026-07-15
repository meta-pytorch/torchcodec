// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "AVIOFileLikeContext.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

AVIOFileLikeContext::AVIOFileLikeContext(
    const py::object& file_like,
    bool is_for_writing)
    : file_like_{UniquePyObject(new py::object(file_like))} {
  {
    // TODO: Is it necessary to acquire the GIL here? Is it maybe even
    // harmful? At the moment, this is only called from within a pybind
    // function, and pybind guarantees we have the GIL.
    py::gil_scoped_acquire gil;

    if (is_for_writing) {
      STD_TORCH_CHECK(
          py::hasattr(file_like, "write"),
          "File like object must implement a write method for writing.");
    } else {
      STD_TORCH_CHECK(
          py::hasattr(file_like, "read"),
          "File like object must implement a read method for reading.");
    }

    STD_TORCH_CHECK(
        py::hasattr(file_like, "seek"),
        "File like object must implement a seek method.");
  }
  create_avio_context(is_for_writing);
}

int AVIOFileLikeContext::read(uint8_t* buf, int size) {
  py::gil_scoped_acquire gil;

  int total_num_read = 0;
  while (total_num_read < size) {
    int request = size - total_num_read;

    // The Python method returns the actual bytes, which we access through
    // the py::bytes wrapper. That wrapper, however, does not provide us
    // access to the underlying data pointer, which we need for the memcpy
    // below. So we convert the bytes to a string_view to get access to
    // the data pointer. Because it's a view and not a copy, it should be
    // cheap.
    auto bytes_read = static_cast<py::bytes>(file_like_->attr("read")(request));
    auto bytes_view = static_cast<std::string_view>(bytes_read);

    int num_bytes_read = static_cast<int>(bytes_view.size());
    if (num_bytes_read == 0) {
      break;
    }

    STD_TORCH_CHECK(
        num_bytes_read <= request,
        "Requested up to ",
        request,
        " bytes but, received ",
        num_bytes_read,
        " bytes. The given object does not conform to read protocol "
        "of file object.");

    std::memcpy(buf, bytes_view.data(), num_bytes_read);
    buf += num_bytes_read;
    total_num_read += num_bytes_read;
  }

  return total_num_read == 0 ? -1 : total_num_read;
}

int AVIOFileLikeContext::write(const uint8_t* buf, int size) {
  py::gil_scoped_acquire gil;
  py::bytes bytes_obj(reinterpret_cast<const char*>(buf), size);
  return py::cast<int>(file_like_->attr("write")(bytes_obj));
}

int64_t AVIOFileLikeContext::seek(int64_t offset, int whence) {
  py::gil_scoped_acquire gil;
  return py::cast<int64_t>(file_like_->attr("seek")(offset, whence));
}

int64_t AVIOFileLikeContext::get_size() {
  // Size of file-like is typically unknown, since the data is potentially
  // streaming.
  return INT64_MAX;
}

} // namespace facebook::torchcodec
