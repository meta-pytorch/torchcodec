// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// nanobind.h pulls in Python.h, so it must come before any standard headers.
#include <nanobind/nanobind.h>

#include <cstdint>

#include "AVIOFileLikeContext.h"

namespace nb = nanobind;

namespace facebook::torchcodec {

// Note: It's not immediately obvous why we need both custom_ops.cpp and
//       pybind_ops.cpp. We do all other Python to C++ bridging in
//       custom_ops.cpp, so why have an explicit nanobind file?
//
//       The reason is that we want to accept OWNERSHIP of a file-like object
//       from the Python side. In order to do that, we need a proper
//       nb::object. For raw bytes, we can launder that through a tensor on the
//       custom_ops.cpp side, but we can't launder a proper Python object
//       through a tensor. Custom ops can't accept a proper Python object
//       through nb::object, so we have to do direct nanobind here.
//
// TODO: Investigate if we can do something better here. See:
//         https://github.com/pytorch/torchcodec/issues/896
//       Short version is that we're laundering a pointer through an int, the
//       Python side forwards that to decoder creation functions in
//       custom_ops.cpp and we do another cast on that side to get a pointer
//       again. We want to investigate if we can do something cleaner by
//       defining proper nanobind objects.
int64_t create_file_like_context(nb::object file_like, bool is_for_writing) {
  AVIOFileLikeContext* context =
      new AVIOFileLikeContext(std::move(file_like), is_for_writing);
  // Launder the *base* AVIOContextHolder pointer (not the derived pointer) so
  // that custom_ops.cpp, which never sees the nanobind-dependent
  // AVIOFileLikeContext type, can cast the int back to an AVIOContextHolder*
  // and delete it through its virtual destructor.
  return reinterpret_cast<int64_t>(static_cast<AVIOContextHolder*>(context));
}

#ifndef PYBIND_OPS_MODULE_NAME
#error PYBIND_OPS_MODULE_NAME must be defined!
#endif

NB_MODULE(PYBIND_OPS_MODULE_NAME, m) {
  m.def("create_file_like_context", &create_file_like_context);
}

} // namespace facebook::torchcodec
