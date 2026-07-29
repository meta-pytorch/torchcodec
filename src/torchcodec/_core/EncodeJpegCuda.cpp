// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "EncodeJpegCuda.h"

#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !TORCHCODEC_ENABLE_NVJPEG

namespace facebook::torchcodec {

void encode_jpeg_cuda(
    [[maybe_unused]] const torch::stable::Tensor& image,
    [[maybe_unused]] int64_t quality,
    [[maybe_unused]] IOInterface& interface) {
  STD_TORCH_CHECK(
      false,
      "encode_jpeg: torchcodec was not compiled with nvJPEG support, so JPEG "
      "images cannot be encoded on a CUDA device. Rebuild torchcodec with "
      "ENABLE_CUDA=1 in an environment where the CUDA toolkit (which provides "
      "nvJPEG) is available. If you see this error in a prebuilt wheel, please "
      "report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

#include "CUDACommon.h"
#include "Cache.h"

namespace facebook::torchcodec {

namespace {

// Encoders are, like the JPEG decoders, expensive to create (~270ms) and cheap
// to keep resident (~40Mb of GPU memory), so we cache a generous number per
// device (same cap as the decoder). Caching gives a ~300x speedup over
// constructing an encoder per call.
constexpr int kMaxCachedEncodersPerDevice = 32;

PerGpuCache<CUDAJpegEncoder>& encoder_cache() {
  // Intentionally leaked (allocated with new, never freed) to avoid calling
  // into CUDA/nvJPEG during static destruction, when the CUDA runtime may
  // already be torn down (same reasoning as NVDECCache): the cached encoders'
  // destructors call nvjpegDestroy, which we must not run at process exit.
  static auto* cache = new PerGpuCache<CUDAJpegEncoder>(
      MAX_CUDA_GPUS, kMaxCachedEncodersPerDevice);
  return *cache;
}

} // namespace

void encode_jpeg_cuda(
    const torch::stable::Tensor& image,
    int64_t quality,
    IOInterface& interface) {
  STD_TORCH_CHECK(
      image.device().is_cuda(),
      "Input tensor must be on a CUDA device, got a tensor on ",
      device_type_name(image.device().type()),
      ".");

  torch::stable::Device device = image.device();
  int device_index = get_device_index(device);
  StableDeviceGuard device_guard(device_index);

  cudaStream_t current_stream = get_current_cuda_stream(device_index);

  PerGpuCache<CUDAJpegEncoder>& cache = encoder_cache();
  std::unique_ptr<CUDAJpegEncoder> encoder = cache.get(device);
  if (encoder == nullptr) {
    encoder = std::make_unique<CUDAJpegEncoder>(device);
  }

  std::vector<uint8_t> encoded =
      encoder->encode_image(image, quality, current_stream);
  cache.add_if_cache_has_capacity(device, std::move(encoder));

  interface.write(encoded.data(), static_cast<int>(encoded.size()));
}

CUDAJpegEncoder::CUDAJpegEncoder(const torch::stable::Device& target_device)
    : target_device_(target_device) {
  StableDeviceGuard device_guard(target_device_.index());

  nvjpegStatus_t status;
  status = nvjpegCreateSimple(&nvjpeg_handle_);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg handle: ",
      status);

  status = nvjpegEncoderStateCreate(
      nvjpeg_handle_, &nvjpeg_enc_state_, /*stream=*/nullptr);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg encoder state: ",
      status);

  status = nvjpegEncoderParamsCreate(
      nvjpeg_handle_, &nvjpeg_enc_params_, /*stream=*/nullptr);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg encoder params: ",
      status);
}

CUDAJpegEncoder::~CUDAJpegEncoder() {
  // Like the JPEG decoder, this destructor only runs when an encoder cannot
  // return to the cache; it's never reached during process teardown, because we
  // leak the entire encoder cache to avoid CUDA teardown issues.
  nvjpegEncoderParamsDestroy(nvjpeg_enc_params_);
  nvjpegEncoderStateDestroy(nvjpeg_enc_state_);
  nvjpegDestroy(nvjpeg_handle_);
}

std::vector<uint8_t> CUDAJpegEncoder::encode_image(
    const torch::stable::Tensor& image,
    int64_t quality,
    cudaStream_t stream) {
  STD_TORCH_CHECK(
      image.scalar_type() == kStableUInt8,
      "Input tensor dtype should be uint8");
  STD_TORCH_CHECK(
      image.dim() == 3, "Input data should be a 3-dimensional tensor");

  const int num_channels = static_cast<int>(image.size(0));
  const int height = static_cast<int>(image.size(1));
  const int width = static_cast<int>(image.size(2));
  STD_TORCH_CHECK(
      num_channels == 3,
      "The number of channels should be 3 (RGB) for JPEG encoding on GPU, "
      "got: ",
      num_channels,
      ". Grayscale encoding is only supported on the CPU for now.");

  // nvJPEG reads each channel as a separate planar buffer, so the input must be
  // contiguous (pitch == width below relies on this).
  const auto input = torch::stable::contiguous(image);

  nvjpegStatus_t status;
  status = nvjpegEncoderParamsSetQuality(
      nvjpeg_enc_params_, static_cast<int>(quality), stream);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to set nvjpeg encoder quality: ",
      status);

  status = nvjpegEncoderParamsSetSamplingFactors(
      nvjpeg_enc_params_, NVJPEG_CSS_444, stream);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to set nvjpeg encoder sampling factors: ",
      status);

  nvjpegImage_t nvjpeg_image;
  for (int c = 0; c < num_channels; ++c) {
    nvjpeg_image.channel[c] =
        torch::stable::select(input, 0, c).mutable_data_ptr<uint8_t>();
    nvjpeg_image.pitch[c] = width;
  }
  for (int c = num_channels; c < NVJPEG_MAX_COMPONENT; ++c) {
    nvjpeg_image.channel[c] = nullptr;
    nvjpeg_image.pitch[c] = 0;
  }

  status = nvjpegEncodeImage(
      nvjpeg_handle_,
      nvjpeg_enc_state_,
      nvjpeg_enc_params_,
      &nvjpeg_image,
      NVJPEG_INPUT_RGB,
      width,
      height,
      stream);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS, "Failed to encode image: ", status);

  // Retrieve the encoded bitstream directly into host memory. First query the
  // length (with a null buffer), then allocate the output and fill it.
  size_t length = 0;
  status = nvjpegEncodeRetrieveBitstream(
      nvjpeg_handle_, nvjpeg_enc_state_, nullptr, &length, stream);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded bitstream size: ",
      status);

  cudaError_t cuda_status = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(
      cuda_status == cudaSuccess,
      "Failed to synchronize CUDA stream: ",
      cuda_status);

  std::vector<uint8_t> output(length);
  status = nvjpegEncodeRetrieveBitstream(
      nvjpeg_handle_, nvjpeg_enc_state_, output.data(), &length, stream);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded bitstream: ",
      status);

  // Host-synchronize before returning: the encoder (and its internal nvJPEG
  // buffers) goes back to the pool and may be reused immediately by the next
  // call, so the copy into `output` must complete first.
  cuda_status = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(
      cuda_status == cudaSuccess,
      "Failed to synchronize CUDA stream: ",
      cuda_status);

  return output;
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_NVJPEG
