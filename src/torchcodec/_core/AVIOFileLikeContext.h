// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// Python.h must be included before any standard headers, so we include it
// first. The CPython *limited* API (a.k.a. stable ABI) is selected via the
// Py_LIMITED_API compile definition set on this target in CMake, which is what
// allows us to ship a single abi3 wheel that works across Python versions.
#include <Python.h>

#include "AVIOContextHolder.h"

namespace facebook::torchcodec {

// Enables uers to pass in a Python file-like object. We then forward all read
// and seek calls back up to the methods on the Python object.
class AVIOFileLikeContext : public AVIOContextHolder {
 public:
  explicit AVIOFileLikeContext(PyObject* file_like, bool is_for_writing);
  ~AVIOFileLikeContext() override;

  int read(uint8_t* buf, int size) override;
  int write(const uint8_t* buf, int size) override;
  int64_t seek(int64_t offset, int whence) override;
  int64_t get_size() override;

 private:
  // We maintain a reference to the file-like object because the file-like
  // object that was created on the Python side must live as long as our
  // potential use. That is, even if there are no more references to the object
  // on the Python side, we require that the object is still live.
  //
  // We must hold the GIL whenever we touch this object's reference count, in
  // particular when we Py_DECREF it in the destructor: the destructor may run
  // from an arbitrary C++ scope that does not necessarily hold the GIL, so the
  // destructor acquires it explicitly. For all of the common pitfalls, see:
  //
  //   https://pybind11.readthedocs.io/en/stable/advanced/misc.html#common-sources-of-global-interpreter-lock-errors
  PyObject* file_like_ = nullptr;
};

} // namespace facebook::torchcodec
