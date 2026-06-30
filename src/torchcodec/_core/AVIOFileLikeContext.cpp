// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "AVIOFileLikeContext.h"

#include <cstring>

#include "StableABICompat.h"

namespace facebook::torchcodec {

namespace {

// RAII wrapper around the GIL, replacing pybind11's py::gil_scoped_acquire.
struct GilGuard {
  GilGuard() : state_(PyGILState_Ensure()) {}

  ~GilGuard() {
    PyGILState_Release(state_);
  }

  GilGuard(const GilGuard&) = delete;
  GilGuard& operator=(const GilGuard&) = delete;

 private:
  PyGILState_STATE state_;
};

// Fetch the current Python error (if any) as a string and clear it. Must be
// called while holding the GIL. We use this to surface Python-side exceptions
// (raised by the file-like object's methods) through STD_TORCH_CHECK.
//
// Note: unlike pybind11, which used to translate a Python exception back into
// the *same* exception type at the torch/Python boundary (e.g. a TypeError
// raised in read() would surface as a TypeError), we cannot do that here: the
// torch boundary only restores pybind11's own error_already_set type, which we
// can't construct under the limited API. So file-like callback errors surface
// as RuntimeError, with the original Python error text included in the message.
//
// We deliberately use only limited-API functions (PyUnicode_AsUTF8String +
// PyBytes_AsString) rather than PyUnicode_AsUTF8, which is not part of the
// stable ABI.
std::string getPythonErrorAndClear() {
  if (PyErr_Occurred() == nullptr) {
    return "";
  }
  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);

  std::string message;
  if (value != nullptr) {
    PyObject* str = PyObject_Str(value);
    if (str != nullptr) {
      PyObject* utf8 = PyUnicode_AsUTF8String(str);
      if (utf8 != nullptr) {
        const char* data = PyBytes_AsString(utf8);
        if (data != nullptr) {
          message = data;
        }
        Py_DECREF(utf8);
      }
      Py_DECREF(str);
    }
  }
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  PyErr_Clear();
  return message;
}

} // namespace

AVIOFileLikeContext::AVIOFileLikeContext(
    PyObject* file_like,
    bool is_for_writing)
    : file_like_(file_like) {
  {
    // TODO: Is it necessary to acquire the GIL here? Is it maybe even
    // harmful? At the moment, this is only called from within a pybind
    // function, and pybind guarantees we have the GIL.
    GilGuard gil;

    // Take ownership of the file-like object: it must outlive any of our
    // potential use of it, even if the Python side drops all its references.
    Py_INCREF(file_like_);

    if (is_for_writing) {
      STD_TORCH_CHECK(
          PyObject_HasAttrString(file_like_, "write") == 1,
          "File like object must implement a write method for writing.");
    } else {
      STD_TORCH_CHECK(
          PyObject_HasAttrString(file_like_, "read") == 1,
          "File like object must implement a read method for reading.");
    }

    STD_TORCH_CHECK(
        PyObject_HasAttrString(file_like_, "seek") == 1,
        "File like object must implement a seek method.");
  }
  create_avio_context(is_for_writing);
}

AVIOFileLikeContext::~AVIOFileLikeContext() {
  // We must hold the GIL when the reference count is updated. See the note on
  // file_like_ in the header.
  GilGuard gil;
  Py_XDECREF(file_like_);
}

int AVIOFileLikeContext::read(uint8_t* buf, int size) {
  GilGuard gil;

  int total_num_read = 0;
  while (total_num_read < size) {
    int request = size - total_num_read;

    // The Python method returns the actual bytes. We access the underlying
    // data pointer (needed for the memcpy below) through the limited-API
    // PyBytes_AsStringAndSize, which gives us a view into the bytes object
    // without copying. Because it's a view and not a copy, it should be cheap.
    PyObject* bytes_read =
        PyObject_CallMethod(file_like_, "read", "i", request);
    STD_TORCH_CHECK(
        bytes_read != nullptr,
        "Call to read() on file like object failed: ",
        getPythonErrorAndClear());

    char* data = nullptr;
    Py_ssize_t num_bytes = 0;
    if (PyBytes_AsStringAndSize(bytes_read, &data, &num_bytes) != 0) {
      std::string error = getPythonErrorAndClear();
      Py_DECREF(bytes_read);
      STD_TORCH_CHECK(
          false, "read() did not return a bytes-like object: ", error);
    }

    int num_bytes_read = static_cast<int>(num_bytes);
    if (num_bytes_read == 0) {
      Py_DECREF(bytes_read);
      break;
    }

    if (num_bytes_read > request) {
      Py_DECREF(bytes_read);
      STD_TORCH_CHECK(
          false,
          "Requested up to ",
          request,
          " bytes but, received ",
          num_bytes_read,
          " bytes. The given object does not conform to read protocol "
          "of file object.");
    }

    std::memcpy(buf, data, num_bytes_read);
    Py_DECREF(bytes_read);
    buf += num_bytes_read;
    total_num_read += num_bytes_read;
  }

  return total_num_read == 0 ? -1 : total_num_read;
}

int AVIOFileLikeContext::write(const uint8_t* buf, int size) {
  GilGuard gil;
  PyObject* bytes_obj =
      PyBytes_FromStringAndSize(reinterpret_cast<const char*>(buf), size);
  STD_TORCH_CHECK(
      bytes_obj != nullptr,
      "Failed to create bytes object: ",
      getPythonErrorAndClear());

  PyObject* result = PyObject_CallMethod(file_like_, "write", "O", bytes_obj);
  Py_DECREF(bytes_obj);
  STD_TORCH_CHECK(
      result != nullptr,
      "Call to write() on file like object failed: ",
      getPythonErrorAndClear());

  long num_bytes_written = PyLong_AsLong(result);
  Py_DECREF(result);
  STD_TORCH_CHECK(
      !(num_bytes_written == -1 && PyErr_Occurred() != nullptr),
      "write() did not return an integer: ",
      getPythonErrorAndClear());
  return static_cast<int>(num_bytes_written);
}

int64_t AVIOFileLikeContext::seek(int64_t offset, int whence) {
  GilGuard gil;
  PyObject* result = PyObject_CallMethod(
      file_like_, "seek", "Li", static_cast<long long>(offset), whence);
  STD_TORCH_CHECK(
      result != nullptr,
      "Call to seek() on file like object failed: ",
      getPythonErrorAndClear());

  long long position = PyLong_AsLongLong(result);
  Py_DECREF(result);
  STD_TORCH_CHECK(
      !(position == -1 && PyErr_Occurred() != nullptr),
      "seek() did not return an integer: ",
      getPythonErrorAndClear());
  return static_cast<int64_t>(position);
}

int64_t AVIOFileLikeContext::get_size() {
  // Size of file-like is typically unknown, since the data is potentially
  // streaming.
  return INT64_MAX;
}

} // namespace facebook::torchcodec
