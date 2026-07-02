// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// nanobind.h pulls in Python.h, so it must come before any standard headers.
// We build this module against nanobind's stable-ABI (limited API) mode so we
// can ship a single abi3 wheel that works across Python versions (>= 3.12).
#include <nanobind/nanobind.h>

#include "AVIOContextHolder.h"

namespace nb = nanobind;

namespace facebook::torchcodec {

// Enables uers to pass in a Python file-like object. We then forward all read
// and seek calls back up to the methods on the Python object.
class AVIOFileLikeContext : public AVIOContextHolder {
 public:
  explicit AVIOFileLikeContext(nb::object file_like, bool is_for_writing);

  int read(uint8_t* buf, int size) override;
  int write(const uint8_t* buf, int size) override;
  int64_t seek(int64_t offset, int whence) override;
  int64_t get_size() override;

 private:
  // Note that we dynamically allocate the Python object because we need to
  // strictly control when its destructor is called. We must hold the GIL
  // when its destructor gets called, as it needs to update the reference
  // count. It's easiest to control that when it's dynamic memory. Otherwise,
  // we'd have to ensure whatever enclosing scope holds the object has the GIL,
  // and that's, at least, hard.
  //
  // We maintain a reference to the file-like object because the file-like
  // object that was created on the Python side must live as long as our
  // potential use. That is, even if there are no more references to the object
  // on the Python side, we require that the object is still live.
  struct PyObjectDeleter {
    inline void operator()(nb::object* obj) const {
      if (obj) {
        nb::gil_scoped_acquire gil;
        delete obj;
      }
    }
  };

  using UniquePyObject = std::unique_ptr<nb::object, PyObjectDeleter>;
  UniquePyObject file_like_;
};

} // namespace facebook::torchcodec
