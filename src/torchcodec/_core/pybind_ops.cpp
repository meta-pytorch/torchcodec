// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Python.h must be included before any standard headers. We build this module
// against the CPython *limited* API (a.k.a. stable ABI), selected via the
// Py_LIMITED_API compile definition set on this target in CMake. This is what
// lets a single abi3 wheel work across Python versions: this is the only true
// CPython extension module in torchcodec (core and custom_ops are loaded via
// torch.ops.load_library and don't touch the CPython API).
#include <Python.h>

#include <cstdint>
#include <exception>
#include <stdexcept>

#include "AVIOFileLikeContext.h"

namespace facebook::torchcodec {

// Note: It's not immediately obvous why we need both custom_ops.cpp and
//       pybind_ops.cpp. We do all other Python to C++ bridging in
//       custom_ops.cpp, so why have an explicit Python-extension file?
//
//       The reason is that we want to accept OWNERSHIP of a file-like object
//       from the Python side. In order to do that, we need a proper
//       PyObject. For raw bytes, we can launder that through a tensor on the
//       custom_ops.cpp side, but we can't launder a proper Python object
//       through a tensor. Custom ops can't accept a proper Python object,
//       so we have to expose a direct CPython extension function here.
//
// TODO: Investigate if we can do something better here. See:
//         https://github.com/pytorch/torchcodec/issues/896
//       Short version is that we're laundering a pointer through an int, the
//       Python side forwards that to decoder creation functions in
//       custom_ops.cpp and we do another cast on that side to get a pointer
//       again. We want to investigate if we can do something cleaner by
//       defining proper Python objects.
namespace {

PyObject* create_file_like_context(PyObject* /*self*/, PyObject* args) {
  PyObject* file_like = nullptr;
  int is_for_writing = 0;
  if (!PyArg_ParseTuple(args, "Op", &file_like, &is_for_writing)) {
    return nullptr;
  }

  // pybind11 used to translate C++ exceptions into Python exceptions for us.
  // Now that this is a hand-written extension function, we must catch them
  // ourselves: letting a C++ exception escape into CPython's C call machinery
  // would call std::terminate. STD_TORCH_CHECK (used by AVIOFileLikeContext)
  // throws std::out_of_range for index errors and std::runtime_error otherwise,
  // which we map to IndexError / RuntimeError to match pybind11's behavior.
  try {
    AVIOFileLikeContext* context =
        new AVIOFileLikeContext(file_like, static_cast<bool>(is_for_writing));
    return PyLong_FromLongLong(reinterpret_cast<int64_t>(context));
  } catch (const std::out_of_range& e) {
    PyErr_SetString(PyExc_IndexError, e.what());
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  } catch (...) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Unknown C++ exception in create_file_like_context");
    return nullptr;
  }
}

PyMethodDef methods[] = {
    {"create_file_like_context",
     create_file_like_context,
     METH_VARARGS,
     "Wrap a Python file-like object and return a pointer to it as an int."},
    {nullptr, nullptr, 0, nullptr}};

#ifndef PYBIND_OPS_MODULE_NAME
#error PYBIND_OPS_MODULE_NAME must be defined!
#endif

#define TORCHCODEC_STRINGIFY(x) #x
#define TORCHCODEC_TOSTRING(x) TORCHCODEC_STRINGIFY(x)

PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    TORCHCODEC_TOSTRING(PYBIND_OPS_MODULE_NAME),
    nullptr, // module docstring
    -1, // size of per-interpreter state, -1 means module keeps state in globals
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

} // namespace

} // namespace facebook::torchcodec

// The init function symbol must be PyInit_<module name>, where the module name
// must match _PYBIND_OPS_MODULE_NAME in
// torchcodec/_internally_replaced_utils.py. We derive both the symbol name and
// the module name string above from the PYBIND_OPS_MODULE_NAME compile
// definition, so there's a single source of truth on the C++ side.
#define TORCHCODEC_PYINIT_CONCAT(name) PyInit_##name
#define TORCHCODEC_PYINIT(name) TORCHCODEC_PYINIT_CONCAT(name)

PyMODINIT_FUNC TORCHCODEC_PYINIT(PYBIND_OPS_MODULE_NAME)(void) {
  return PyModule_Create(&facebook::torchcodec::moduledef);
}
