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
#include "MetadataJson.h"
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

// Self-describing DLPack frame: exposes the standard __dlpack__ /
// __dlpack_device__ protocol so numpy / cupy / jax / torch consume it zero-copy
// and learn the correct device (CPU vs CUDA) WITHOUT the Python side having to
// hardcode it. __dlpack__ returns the underlying "dltensor" capsule (single
// consumption, as the protocol requires).
struct DLPackFrame {
  py::object capsule;
  int deviceType = 0;
  int deviceId = 0;
};

py::object frameToDLPackCapsule(const tc::Tensor& data) {
  DLManagedTensor* managed = tc::toDLPack(data);
  py::object capsule = py::reinterpret_steal<py::object>(
      PyCapsule_New(managed, "dltensor", dlpackCapsuleDeleter));
  DLPackFrame frame;
  frame.capsule = std::move(capsule);
  frame.deviceType = static_cast<int>(managed->dl_tensor.device.device_type);
  frame.deviceId = static_cast<int>(managed->dl_tensor.device.device_id);
  return py::cast(frame);
}

SingleStreamDecoder* asDecoder(int64_t decoderPtr) {
  return reinterpret_cast<SingleStreamDecoder*>(decoderPtr);
}

tc::Tensor makeInt64Tensor(const std::vector<int64_t>& values) {
  tc::Tensor t = tc::empty({static_cast<int64_t>(values.size())}, tc::kInt64);
  int64_t* data = t.mutable_data_ptr<int64_t>();
  for (size_t i = 0; i < values.size(); ++i) {
    data[i] = values[i];
  }
  return t;
}

tc::Tensor makeFloat64Tensor(const std::vector<double>& values) {
  tc::Tensor t = tc::empty({static_cast<int64_t>(values.size())}, tc::kFloat64);
  double* data = t.mutable_data_ptr<double>();
  for (size_t i = 0; i < values.size(); ++i) {
    data[i] = values[i];
  }
  return t;
}

py::tuple batchToTuple(FrameBatchOutput& out) {
  return py::make_tuple(
      frameToDLPackCapsule(out.data),
      frameToDLPackCapsule(out.ptsSeconds),
      frameToDLPackCapsule(out.durationSeconds));
}

} // namespace

int64_t create_decoder(const std::string& filename) {
  auto decoder =
      std::make_unique<SingleStreamDecoder>(filename, SeekMode::approximate);
  return reinterpret_cast<int64_t>(decoder.release());
}

void add_video_stream(
    int64_t decoderPtr,
    const std::string& dimensionOrder,
    int64_t streamIndex,
    std::optional<int64_t> numThreads,
    const std::string& device,
    const std::string& deviceVariant) {
  VideoStreamOptions options;
  options.dimensionOrder = dimensionOrder;
  if (numThreads.has_value()) {
    options.ffmpegThreadCount = static_cast<int>(numThreads.value());
  }
  options.device = tc::Device(device);
  options.deviceVariant = deviceVariant;
  std::vector<Transform*> transforms;
  asDecoder(decoderPtr)
      ->addVideoStream(static_cast<int>(streamIndex), transforms, options);
}

void scan_all_streams(int64_t decoderPtr) {
  asDecoder(decoderPtr)->scanFileAndUpdateMetadataAndIndex();
}

py::object get_next_frame(int64_t decoderPtr) {
  FrameOutput frameOutput = asDecoder(decoderPtr)->getNextFrame();
  return frameToDLPackCapsule(frameOutput.data);
}

// Single-frame ops return (data_capsule, pts_seconds, duration_seconds); the
// pts/duration are plain Python floats (no need for the 0-dim tensor form the
// torch.compile path uses).
py::tuple get_frame_at_index(int64_t decoderPtr, int64_t frameIndex) {
  FrameOutput frameOutput = asDecoder(decoderPtr)->getFrameAtIndex(frameIndex);
  return py::make_tuple(
      frameToDLPackCapsule(frameOutput.data),
      frameOutput.ptsSeconds,
      frameOutput.durationSeconds);
}

py::tuple get_frame_played_at(int64_t decoderPtr, double seconds) {
  FrameOutput frameOutput = asDecoder(decoderPtr)->getFramePlayedAt(seconds);
  return py::make_tuple(
      frameToDLPackCapsule(frameOutput.data),
      frameOutput.ptsSeconds,
      frameOutput.durationSeconds);
}

// Batch ops return (data_capsule, pts_capsule, duration_capsule); pts/duration
// are 1-D arrays.
py::tuple get_frames_in_range(
    int64_t decoderPtr,
    int64_t start,
    int64_t stop,
    int64_t step) {
  FrameBatchOutput out =
      asDecoder(decoderPtr)
          ->getFramesInRange(start, stop, step <= 0 ? 1 : step);
  return batchToTuple(out);
}

py::tuple get_frames_at_indices(
    int64_t decoderPtr,
    const std::vector<int64_t>& frameIndices) {
  FrameBatchOutput out =
      asDecoder(decoderPtr)->getFramesAtIndices(makeInt64Tensor(frameIndices));
  return batchToTuple(out);
}

py::tuple get_frames_by_pts(
    int64_t decoderPtr,
    const std::vector<double>& timestamps) {
  FrameBatchOutput out =
      asDecoder(decoderPtr)->getFramesPlayedAt(makeFloat64Tensor(timestamps));
  return batchToTuple(out);
}

py::tuple get_frames_by_pts_in_range(
    int64_t decoderPtr,
    double startSeconds,
    double stopSeconds,
    std::optional<double> fps) {
  FrameBatchOutput out =
      asDecoder(decoderPtr)
          ->getFramesPlayedInRange(startSeconds, stopSeconds, fps);
  return batchToTuple(out);
}

py::object get_key_frame_indices(int64_t decoderPtr) {
  return frameToDLPackCapsule(asDecoder(decoderPtr)->getKeyFrameIndices());
}

std::string get_json_metadata(int64_t decoderPtr) {
  return getVideoJsonMetadata(asDecoder(decoderPtr));
}

std::string get_container_json_metadata(int64_t decoderPtr) {
  return getContainerJsonMetadata(asDecoder(decoderPtr));
}

std::string get_stream_json_metadata(int64_t decoderPtr, int64_t streamIndex) {
  return getStreamJsonMetadata(asDecoder(decoderPtr), streamIndex);
}

void destroy_decoder(int64_t decoderPtr) {
  delete asDecoder(decoderPtr);
}

#ifndef PYBIND_OPS_MODULE_NAME
#error PYBIND_OPS_MODULE_NAME must be defined!
#endif

PYBIND11_MODULE(PYBIND_OPS_MODULE_NAME, m) {
  // Self-describing DLPack frame returned by the decode ops (see DLPackFrame).
  py::class_<DLPackFrame>(m, "_DLPackFrame")
      .def(
          "__dlpack__",
          [](DLPackFrame& self, py::args, py::kwargs) { return self.capsule; })
      .def("__dlpack_device__", [](DLPackFrame& self) {
        return py::make_tuple(self.deviceType, self.deviceId);
      });

  m.def("create_file_like_context", &create_file_like_context);
  // Torch-free decoding ops.
  m.def("create_decoder", &create_decoder);
  m.def(
      "add_video_stream",
      &add_video_stream,
      py::arg("decoder"),
      py::arg("dimension_order") = "NCHW",
      py::arg("stream_index") = -1,
      py::arg("num_threads") = py::none(),
      py::arg("device") = "cpu",
      py::arg("device_variant") = "default");
  m.def("scan_all_streams", &scan_all_streams);
  m.def("get_next_frame", &get_next_frame);
  m.def("get_frame_at_index", &get_frame_at_index);
  m.def("get_frame_played_at", &get_frame_played_at);
  m.def("get_frames_in_range", &get_frames_in_range);
  m.def("get_frames_at_indices", &get_frames_at_indices);
  m.def("get_frames_by_pts", &get_frames_by_pts);
  m.def(
      "get_frames_by_pts_in_range",
      &get_frames_by_pts_in_range,
      py::arg("decoder"),
      py::arg("start_seconds"),
      py::arg("stop_seconds"),
      py::arg("fps") = py::none());
  m.def("get_key_frame_indices", &get_key_frame_indices);
  m.def("get_json_metadata", &get_json_metadata);
  m.def("get_container_json_metadata", &get_container_json_metadata);
  m.def("get_stream_json_metadata", &get_stream_json_metadata);
  m.def("destroy_decoder", &destroy_decoder);
}

} // namespace facebook::torchcodec
