// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>

#include <tuple>
#include <utility>
#include <vector>

#include "ImageCommon.h"
#include "StableABICompat.h"

#if TORCHCODEC_ENABLE_NVJPEG
#include <cuda_runtime.h>
#include <nvjpeg.h>
#endif

namespace facebook::torchcodec {

FORCE_PUBLIC_VISIBILITY std::vector<torch::stable::Tensor> decode_jpegs_cuda(
    std::vector<torch::stable::Tensor> encoded_images,
    int64_t mode,
    torch::stable::Device device);

#if TORCHCODEC_ENABLE_NVJPEG

class CUDAJpegDecoder {
 public:
  explicit CUDAJpegDecoder(const torch::stable::Device& target_device);
  ~CUDAJpegDecoder();

  std::vector<torch::stable::Tensor> decode_images(
      const std::vector<torch::stable::Tensor>& encoded_images,
      ImageReadMode mode,
      cudaStream_t stream);

 private:
  std::tuple<torch::stable::Tensor, nvjpegImage_t, nvjpegOutputFormat_t>
  allocate_output(
      const torch::stable::Tensor& encoded_image,
      ImageReadMode mode);

  std::pair<std::vector<size_t>, std::vector<size_t>> split_images_by_backend(
      const std::vector<torch::stable::Tensor>& encoded_images);

  void decode_batched_hardware(
      const std::vector<torch::stable::Tensor>& encoded_images,
      const std::vector<size_t>& indices,
      ImageReadMode mode,
      cudaStream_t stream,
      std::vector<torch::stable::Tensor>& output_tensors);

  void decode_software(
      const std::vector<torch::stable::Tensor>& encoded_images,
      const std::vector<size_t>& indices,
      ImageReadMode mode,
      cudaStream_t stream,
      std::vector<torch::stable::Tensor>& output_tensors);

  const torch::stable::Device target_device_;

  // The nvJPEG library handle, and whether this GPU has the fixed-function HW
  // JPEG engine. Both used everywhere.
  nvjpegHandle_t nvjpeg_handle_;
  bool hw_decode_available_{false};

  // HW path only
  nvjpegJpegState_t nvjpeg_state_hw_;
  nvjpegJpegStream_t nvjpeg_stream_;

  // SW path, using the decoupled (host / transfer / device) API. Always
  // created, even when the HW engine is available, because not all JPEGs are
  // supported by the HW engine. Consecutive images alternate between the two
  // pinned buffers and jpeg streams. See decode_software().
  nvjpegJpegDecoder_t sw_decoder_;
  nvjpegJpegState_t sw_decoder_state_;
  nvjpegDecodeParams_t sw_decode_params_;
  nvjpegBufferDevice_t sw_device_buffer_;
  static constexpr int kNumSwPinnedBuffers = 2;
  nvjpegBufferPinned_t sw_pinned_buffers_[kNumSwPinnedBuffers];
  nvjpegJpegStream_t sw_jpeg_streams_[kNumSwPinnedBuffers];
};

#endif // TORCHCODEC_ENABLE_NVJPEG

} // namespace facebook::torchcodec
