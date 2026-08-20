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

void sync_streams(cudaStream_t running_stream, cudaStream_t waiting_stream) {
  if (running_stream == waiting_stream) {
    return;
  }

  cudaEvent_t event;
  cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "cudaEventCreateWithFlags failed: ",
      cudaGetErrorString(err));

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

CudaEvent::~CudaEvent() {
  if (event_ != nullptr) {
    // Destroying an event that hasn't completed yet is fine: CUDA frees it once
    // it does.
    cudaEventDestroy(event_);
  }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
  other.event_ = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
  if (this != &other) {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
    event_ = other.event_;
    other.event_ = nullptr;
  }
  return *this;
}

void CudaEvent::record(cudaStream_t running_stream) {
  if (event_ == nullptr) {
    cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    STD_TORCH_CHECK(
        err == cudaSuccess,
        "cudaEventCreateWithFlags failed: ",
        cudaGetErrorString(err));
  }
  cudaError_t err = cudaEventRecord(event_, running_stream);
  STD_TORCH_CHECK(
      err == cudaSuccess, "cudaEventRecord failed: ", cudaGetErrorString(err));
}

void CudaEvent::make_stream_wait(cudaStream_t waiting_stream) const {
  if (event_ == nullptr) {
    return;
  }
  // The wait captures the event's state as of *now*, so re-recording the event
  // afterwards doesn't retroactively change what `waiting_stream` waits for.
  cudaError_t err = cudaStreamWaitEvent(waiting_stream, event_, 0);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "cudaStreamWaitEvent failed: ",
      cudaGetErrorString(err));
}

void CudaEvent::synchronize() const {
  if (event_ == nullptr) {
    return;
  }
  cudaError_t err = cudaEventSynchronize(event_);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "cudaEventSynchronize failed: ",
      cudaGetErrorString(err));
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
