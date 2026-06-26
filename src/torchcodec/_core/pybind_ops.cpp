// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// pybind11 headers from torch error out if TORCH_TARGET_VERSION is defined,
// so we temporarily undefine it.
// See https://github.com/pytorch/pytorch/pull/174372 for context
#pragma push_macro("TORCH_TARGET_VERSION")
#undef TORCH_TARGET_VERSION
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma pop_macro("TORCH_TARGET_VERSION")
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "AVIOFileLikeContext.h"
#include "SingleStreamDecoder.h"
#include "TCTensor.h"
#include "Transform.h"

namespace py = pybind11;

namespace facebook::torchcodec {

// Note: It's not immediately obvous why we need both custom_ops.cpp and
//       pybind_ops.cpp. We do all other Python to C++ bridging in
//       custom_ops.cpp, and that even depends on pybind11, so why have an
//       explicit pybind-only file?
//
//       The reason is that we want to accept OWNERSHIP of a file-like object
//       from the Python side. In order to do that, we need a proper
//       py::object. For raw bytes, we can launder that through a tensor on the
//       custom_ops.cpp side, but we can't launder a proper Python object
//       through a tensor. Custom ops can't accept a proper Python object
//       through py::object, so we have to do direct pybind11 here.
//
// TODO: Investigate if we can do something better here. See:
//         https://github.com/pytorch/torchcodec/issues/896
//       Short version is that we're laundering a pointer through an int, the
//       Python side forwards that to decoder creation functions in
//       custom_ops.cpp and we do another cast on that side to get a pointer
//       again. We want to investigate if we can do something cleaner by
//       defining proper pybind objects.
int64_t create_file_like_context(py::object file_like, bool is_for_writing) {
  AVIOFileLikeContext* context =
      new AVIOFileLikeContext(file_like, is_for_writing);
  return reinterpret_cast<int64_t>(context);
}

// ---------------------------------------------------------------------------
// Torch-free decoding ops.
//
// These expose enough of SingleStreamDecoder to decode frames WITHOUT PyTorch.
// Decoded frames are returned as DLPack capsules (PyCapsule named "dltensor")
// which numpy / cupy / jax consume zero-copy via from_dlpack. The decoder is
// passed across the boundary as an opaque int64 pointer (same laundering style
// as create_file_like_context).
// ---------------------------------------------------------------------------

namespace {

// Frees the DLManagedTensor if the capsule was never consumed (a consumer
// renames it to "used_dltensor" and takes over the deleter).
void dlpackCapsuleDeleter(PyObject* capsule) {
  if (PyCapsule_IsValid(capsule, "dltensor")) {
    auto* managed = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule, "dltensor"));
    if (managed != nullptr && managed->deleter != nullptr) {
      managed->deleter(managed);
    }
  }
}

py::object frameToDLPackCapsule(const tc::Tensor& data) {
  DLManagedTensor* managed = tc::toDLPack(data);
  return py::reinterpret_steal<py::object>(
      PyCapsule_New(managed, "dltensor", dlpackCapsuleDeleter));
}

SingleStreamDecoder* asDecoder(int64_t decoderPtr) {
  return reinterpret_cast<SingleStreamDecoder*>(decoderPtr);
}

} // namespace

int64_t create_decoder(const std::string& filename) {
  auto decoder =
      std::make_unique<SingleStreamDecoder>(filename, SeekMode::approximate);
  return reinterpret_cast<int64_t>(decoder.release());
}

void add_video_stream(int64_t decoderPtr, const std::string& dimensionOrder) {
  VideoStreamOptions options;
  options.dimensionOrder = dimensionOrder;
  std::vector<Transform*> transforms;
  asDecoder(decoderPtr)->addVideoStream(-1, transforms, options);
}

py::object get_next_frame(int64_t decoderPtr) {
  FrameOutput frameOutput = asDecoder(decoderPtr)->getNextFrame();
  return frameToDLPackCapsule(frameOutput.data);
}

void destroy_decoder(int64_t decoderPtr) {
  delete asDecoder(decoderPtr);
}

#ifndef PYBIND_OPS_MODULE_NAME
#error PYBIND_OPS_MODULE_NAME must be defined!
#endif

PYBIND11_MODULE(PYBIND_OPS_MODULE_NAME, m) {
  m.def("create_file_like_context", &create_file_like_context);
  // Torch-free decoding ops.
  m.def("create_decoder", &create_decoder);
  m.def("add_video_stream", &add_video_stream, py::arg("decoder"),
        py::arg("dimension_order") = "NCHW");
  m.def("get_next_frame", &get_next_frame);
  m.def("destroy_decoder", &destroy_decoder);
}

} // namespace facebook::torchcodec
