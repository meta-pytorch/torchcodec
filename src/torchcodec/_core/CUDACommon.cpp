// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "CUDACommon.h"
#include "Frame.h"
#include "StableABICompat.h"
#include "ValidationUtils.h"

namespace facebook::torchcodec {

// Make waitingStream wait until all work currently enqueued on runningStream
// has completed.
// TODO_API_BREAKDOWN PERF P2: this creates and destroys a cudaEvent_t on every
// call, and a timing-enabled one at that, which is the more expensive flavour.
// It sits on the per-frame path: convert_yuv_frame_to_rgb() calls it for every
// single frame, including when both streams are the same and the whole thing
// is a no-op. Two easy wins: return early when the two streams are equal, and
// take a caller-owned event created with cudaEventDisableTiming instead of
// allocating one here (see record_surface_read() in BetaCudaDeviceInterface).
void sync_streams(cudaStream_t running_stream, cudaStream_t waiting_stream) {
  cudaEvent_t event;
  cudaError_t err = cudaEventCreate(&event);
  STD_TORCH_CHECK(
      err == cudaSuccess, "cudaEventCreate failed: ", cudaGetErrorString(err));

  err = cudaEventRecord(event, running_stream);
  STD_TORCH_CHECK(
      err == cudaSuccess, "cudaEventRecord failed: ", cudaGetErrorString(err));

  err = cudaStreamWaitEvent(waiting_stream, event, 0);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "cudaStreamWaitEvent failed: ",
      cudaGetErrorString(err));

  cudaEventDestroy(event);
}

void initialize_cuda_context_with_pytorch(const StableDevice& device) {
  // It is important for pytorch itself to create the cuda context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the cuda context.
  torch::stable::Tensor dummy_tensor_for_cuda_initialization =
      torch::stable::empty(
          {1}, kStableUInt8, std::nullopt, StableDevice(device));
  torch::stable::zero_(dummy_tensor_for_cuda_initialization);
}

void validate_pre_allocated_tensor_shape(
    const std::optional<torch::stable::Tensor>& pre_allocated_output_tensor,
    const FrameDims& frame_dims) {
  if (pre_allocated_output_tensor.has_value()) {
    auto shape = pre_allocated_output_tensor.value().sizes();
    STD_TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == frame_dims.height) &&
            (shape[1] == frame_dims.width) && (shape[2] == 3),
        "Expected tensor of shape ",
        frame_dims.height,
        "x",
        frame_dims.width,
        "x3, got ",
        int_array_ref_to_string(shape));
  }
}

} // namespace facebook::torchcodec
