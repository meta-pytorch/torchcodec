// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "AVIOFileLikeContext.h"

#include <cstring>
#include <stdexcept>

#include "StableABICompat.h"

namespace facebook::torchcodec {

namespace {

// nanobind throws nb::python_error when a Python call fails. These read/write/
// seek methods are invoked as FFmpeg AVIO callbacks, so such an exception would
// propagate through FFmpeg's C frames up to the torch op boundary. We catch it
// here, while we still hold the GIL, and rethrow a plain std::runtime_error
// with the original message: that crosses the C frames and the torch boundary
// cleanly. Note that the original Python exception *type* (e.g. TypeError)
// cannot be preserved -- torch only restores pybind11's own error type, not
// nanobind's -- so these surface as RuntimeError with the original message.
//
// We format the message ourselves with str(exc) via the stable-ABI C API,
// rather than nb::python_error::what(), because under the limited API the
// latter formats a full traceback and can fail (yielding an opaque
// "<error while formatting exception>").
[[noreturn]] void rethrowPythonError(
    const char* method,
    const nb::python_error& e) {
  std::string detail = "<unknown Python error>";
  PyObject* str = PyObject_Str(e.value().ptr());
  if (str != nullptr) {
    PyObject* utf8 = PyUnicode_AsUTF8String(str);
    if (utf8 != nullptr) {
      const char* data = PyBytes_AsString(utf8);
      if (data != nullptr) {
        detail = data;
      }
      Py_DECREF(utf8);
    }
    Py_DECREF(str);
  }
  PyErr_Clear();
  throw std::runtime_error(
      std::string("Error in the file like object's ") + method +
      "() method: " + detail);
}

} // namespace

AVIOFileLikeContext::AVIOFileLikeContext(
    nb::object file_like,
    bool is_for_writing)
    : file_like_{UniquePyObject(new nb::object(std::move(file_like)))} {
  {
    // TODO: Is it necessary to acquire the GIL here? Is it maybe even
    // harmful? At the moment, this is only called from within a nanobind
    // function, and nanobind guarantees we have the GIL.
    nb::gil_scoped_acquire gil;

    if (is_for_writing) {
      STD_TORCH_CHECK(
          nb::hasattr(*file_like_, "write"),
          "File like object must implement a write method for writing.");
    } else {
      STD_TORCH_CHECK(
          nb::hasattr(*file_like_, "read"),
          "File like object must implement a read method for reading.");
    }

    STD_TORCH_CHECK(
        nb::hasattr(*file_like_, "seek"),
        "File like object must implement a seek method.");
  }
  create_avio_context(is_for_writing);
}

int AVIOFileLikeContext::read(uint8_t* buf, int size) {
  nb::gil_scoped_acquire gil;

  try {
    int total_num_read = 0;
    while (total_num_read < size) {
      int request = size - total_num_read;

      // The Python method returns the actual bytes, which we access through the
      // nb::bytes wrapper. We use its data pointer for the memcpy below; since
      // it's a view into the bytes object and not a copy, it should be cheap.
      nb::object result = file_like_->attr("read")(request);
      nb::bytes bytes_read(result);

      int num_bytes_read = static_cast<int>(bytes_read.size());
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

      std::memcpy(buf, bytes_read.data(), num_bytes_read);
      buf += num_bytes_read;
      total_num_read += num_bytes_read;
    }

    return total_num_read == 0 ? -1 : total_num_read;
  } catch (const nb::python_error& e) {
    rethrowPythonError("read", e);
  }
}

int AVIOFileLikeContext::write(const uint8_t* buf, int size) {
  nb::gil_scoped_acquire gil;
  try {
    nb::bytes bytes_obj(buf, static_cast<size_t>(size));
    return nb::cast<int>(file_like_->attr("write")(bytes_obj));
  } catch (const nb::python_error& e) {
    rethrowPythonError("write", e);
  }
}

int64_t AVIOFileLikeContext::seek(int64_t offset, int whence) {
  nb::gil_scoped_acquire gil;
  try {
    return nb::cast<int64_t>(file_like_->attr("seek")(offset, whence));
  } catch (const nb::python_error& e) {
    rethrowPythonError("seek", e);
  }
}

int64_t AVIOFileLikeContext::get_size() {
  // Size of file-like is typically unknown, since the data is potentially
  // streaming.
  return INT64_MAX;
}

} // namespace facebook::torchcodec
