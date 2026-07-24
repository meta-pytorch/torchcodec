// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>

#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

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

// Owns all the nvJPEG handles/state/buffers for one GPU device. Reused across
// calls via NVJpegCache (see DecodeJpegCuda.cpp). The decode runs on the CUDA
// stream passed to decode_images (the caller's current stream), which the
// decoder does not own -- so it honors torch.cuda.Stream() and each call can
// use a different stream.
class CUDAJpegDecoder {
 public:
  explicit CUDAJpegDecoder(const torch::stable::Device& target_device);
  ~CUDAJpegDecoder();

  std::vector<torch::stable::Tensor> decode_images(
      const std::vector<torch::stable::Tensor>& encoded_images,
      const nvjpegOutputFormat_t& output_format,
      cudaStream_t stream);

 private:
  std::tuple<
      std::vector<nvjpegImage_t>,
      std::vector<torch::stable::Tensor>,
      std::vector<int>>
  prepare_buffers(
      const std::vector<torch::stable::Tensor>& encoded_images,
      const nvjpegOutputFormat_t& output_format);

  const torch::stable::Device target_device_;
  nvjpegJpegState_t nvjpeg_state_;
  nvjpegJpegState_t nvjpeg_decoupled_state_;
  nvjpegBufferPinned_t pinned_buffers_[2];
  nvjpegBufferDevice_t device_buffer_;
  nvjpegJpegStream_t jpeg_streams_[2];
  nvjpegDecodeParams_t nvjpeg_decode_params_;
  nvjpegJpegDecoder_t nvjpeg_decoder_;
  bool hw_decode_available_{false};
  nvjpegHandle_t nvjpeg_handle_;
};

// A genuine per-device pool of reusable CUDAJpegDecoder objects. This replaces
// torchvision's single global decoder + coarse mutex (which was keyed only on
// device and rebuilt on every device switch): there is one pool instance per
// GPU, so switching devices no longer destroys and recreates the decoder, and
// concurrent callers each take their own decoder instead of serializing on one.
// Modeled on NVDECCache.
class NVJpegCache {
 public:
  static NVJpegCache& get_cache(const torch::stable::Device& device);

  // Take a decoder from the pool, or create a fresh one if the pool is empty.
  std::unique_ptr<CUDAJpegDecoder> get_decoder(
      const torch::stable::Device& device);

  // Return a decoder to the pool for reuse (dropped if the pool is full).
  void return_decoder(std::unique_ptr<CUDAJpegDecoder> decoder);

 private:
  static NVJpegCache* get_cache_instances();

  std::vector<std::unique_ptr<CUDAJpegDecoder>> pool_;
  std::mutex pool_lock_;
};

#endif // TORCHCODEC_ENABLE_NVJPEG

} // namespace facebook::torchcodec
