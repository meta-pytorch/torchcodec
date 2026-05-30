// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "AVIOFileLikeContext.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

AVIOFileLikeContext::AVIOFileLikeContext(
    const py::object& fileLike,
    bool isForWriting)
    : fileLike_{UniquePyObject(new py::object(fileLike))} {
  {
    // TODO: Is it necessary to acquire the GIL here? Is it maybe even
    // harmful? At the moment, this is only called from within a pybind
    // function, and pybind guarantees we have the GIL.
    py::gil_scoped_acquire gil;

    if (isForWriting) {
      STD_TORCH_CHECK(
          py::hasattr(fileLike, "write"),
          "File like object must implement a write method for writing.");
    } else {
      STD_TORCH_CHECK(
          py::hasattr(fileLike, "read"),
          "File like object must implement a read method for reading.");
    }

    STD_TORCH_CHECK(
        py::hasattr(fileLike, "seek"),
        "File like object must implement a seek method.");
  }
  createAVIOContext(isForWriting);
}

int AVIOFileLikeContext::read(uint8_t* buf, int size) {
  py::gil_scoped_acquire gil;

  int totalNumRead = 0;
  while (totalNumRead < size) {
    int request = size - totalNumRead;

    // The Python method returns the actual bytes, which we access through
    // the py::bytes wrapper. That wrapper, however, does not provide us
    // access to the underlying data pointer, which we need for the memcpy
    // below. So we convert the bytes to a string_view to get access to
    // the data pointer. Because it's a view and not a copy, it should be
    // cheap.
    auto bytesRead = static_cast<py::bytes>(fileLike_->attr("read")(request));
    auto bytesView = static_cast<std::string_view>(bytesRead);

    int numBytesRead = static_cast<int>(bytesView.size());
    if (numBytesRead == 0) {
      break;
    }

    STD_TORCH_CHECK(
        numBytesRead <= request,
        "Requested up to ",
        request,
        " bytes but, received ",
        numBytesRead,
        " bytes. The given object does not conform to read protocol "
        "of file object.");

    std::memcpy(buf, bytesView.data(), numBytesRead);
    buf += numBytesRead;
    totalNumRead += numBytesRead;
  }

  return totalNumRead == 0 ? -1 : totalNumRead;
}

int AVIOFileLikeContext::write(const uint8_t* buf, int size) {
  py::gil_scoped_acquire gil;
  py::bytes bytes_obj(reinterpret_cast<const char*>(buf), size);
  return py::cast<int>(fileLike_->attr("write")(bytes_obj));
}

int64_t AVIOFileLikeContext::seek(int64_t offset, int whence) {
  py::gil_scoped_acquire gil;
  return py::cast<int64_t>(fileLike_->attr("seek")(offset, whence));
}

int64_t AVIOFileLikeContext::getSize() {
  // Size of file-like is typically unknown, since the data is potentially
  // streaming.
  return INT64_MAX;
}

} // namespace facebook::torchcodec
