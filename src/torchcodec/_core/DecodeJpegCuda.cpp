// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecodeJpegCuda.h"

#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !TORCHCODEC_ENABLE_NVJPEG

namespace facebook::torchcodec {

std::vector<torch::stable::Tensor> decode_jpegs_cuda(
    [[maybe_unused]] std::vector<torch::stable::Tensor> encoded_images,
    [[maybe_unused]] int64_t mode,
    [[maybe_unused]] torch::stable::Device device) {
  STD_TORCH_CHECK(
      false,
      "decode_jpeg: torchcodec was not compiled with nvJPEG support, so JPEG "
      "images cannot be decoded on a CUDA device. Rebuild torchcodec with "
      "ENABLE_CUDA=1 in an environment where the CUDA toolkit (which provides "
      "nvJPEG) is available. If you see this error in a prebuilt wheel, please "
      "report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

#include <torch/csrc/stable/c/shim.h>

#include "Exif.h"
#include "ImageCommon.h"

namespace facebook::torchcodec {

using namespace exif_private;

namespace {

// How many idle decoders to keep around per GPU. Small: each decoder holds
// nvJPEG handles plus pinned/device buffers, and calls are usually serial so
// one gets reused; a few slots let concurrent callers avoid rebuilding.
constexpr size_t kMaxCachedDecodersPerDevice = 4;

// PyTorch supports up to 128 GPUs (see CUDACommon.h's MAX_CUDA_GPUS). Kept
// local so the FFmpeg-free image library doesn't need to include CUDACommon.h.
constexpr int kMaxCudaGpus = 128;

// Resolve a concrete device index, turning the "current device" (-1) into a
// real index. Mirrors get_device_index() in CUDACommon.cpp.
int resolve_device_index(const torch::stable::Device& device) {
  int device_index = static_cast<int>(device.index());
  STD_TORCH_CHECK(
      device_index >= -1 && device_index < kMaxCudaGpus,
      "Invalid device index = ",
      device_index);
  if (device_index == -1) {
    STD_TORCH_CHECK(
        cudaGetDevice(&device_index) == cudaSuccess,
        "Failed to get current CUDA device.");
  }
  return device_index;
}

nvjpegOutputFormat_t output_format_from_mode(int64_t mode) {
  switch (static_cast<ImageReadMode>(mode)) {
    case ImageReadMode::UNCHANGED:
      // NVJPEG_OUTPUT_UNCHANGED yields differently-sized channels depending on
      // subsampling, so we decode as RGB and slice grayscale images back down
      // to a single channel later (matching torchvision).
      return NVJPEG_OUTPUT_UNCHANGED;
    case ImageReadMode::GRAY:
      return NVJPEG_OUTPUT_Y;
    case ImageReadMode::RGB:
      return NVJPEG_OUTPUT_RGB;
    default:
      STD_TORCH_CHECK(
          false,
          "The provided mode is not supported for JPEG decoding on GPU. "
          "nvJPEG natively supports UNCHANGED, GRAY and RGB; alpha modes are "
          "emulated in Python.");
  }
}

} // namespace

NVJpegCache* NVJpegCache::get_cache_instances() {
  // Intentionally leaked to avoid calling into CUDA/nvJPEG during static
  // destruction, when the CUDA runtime may already be torn down (same reasoning
  // as NVDECCache).
  static NVJpegCache* cache_instances = new NVJpegCache[kMaxCudaGpus];
  return cache_instances;
}

NVJpegCache& NVJpegCache::get_cache(const torch::stable::Device& device) {
  return get_cache_instances()[resolve_device_index(device)];
}

std::unique_ptr<CUDAJpegDecoder> NVJpegCache::get_decoder(
    const torch::stable::Device& device) {
  {
    std::lock_guard<std::mutex> lock(pool_lock_);
    if (!pool_.empty()) {
      auto decoder = std::move(pool_.back());
      pool_.pop_back();
      return decoder;
    }
  }
  // Create outside the lock: constructing nvJPEG state is relatively expensive
  // and doesn't need the pool.
  return std::make_unique<CUDAJpegDecoder>(device);
}

void NVJpegCache::return_decoder(std::unique_ptr<CUDAJpegDecoder> decoder) {
  STD_TORCH_CHECK(decoder != nullptr, "decoder must not be null");
  std::lock_guard<std::mutex> lock(pool_lock_);
  if (pool_.size() < kMaxCachedDecodersPerDevice) {
    pool_.push_back(std::move(decoder));
  }
  // Otherwise let `decoder` go out of scope and be destroyed.
}

std::vector<torch::stable::Tensor> decode_jpegs_cuda(
    std::vector<torch::stable::Tensor> encoded_images,
    int64_t mode,
    torch::stable::Device device) {
  STD_TORCH_CHECK(
      device.is_cuda(), "Expected the device parameter to be a cuda device");
  STD_TORCH_CHECK(
      !encoded_images.empty(), "Expected at least one image to decode");

  std::vector<torch::stable::Tensor> contig_images;
  contig_images.reserve(encoded_images.size());
  std::vector<ExifOrientation> orientations;
  orientations.reserve(encoded_images.size());

  for (const auto& encoded_image : encoded_images) {
    STD_TORCH_CHECK(
        encoded_image.scalar_type() == torch::headeronly::ScalarType::Byte,
        "Expected a torch.uint8 tensor");
    STD_TORCH_CHECK(
        !encoded_image.is_cuda(),
        "The input tensor must be on CPU when decoding with nvjpeg");
    STD_TORCH_CHECK(
        encoded_image.dim() == 1 && encoded_image.numel() > 0,
        "Expected a non empty 1-dimensional tensor");

    // nvjpeg requires images to be contiguous.
    auto contig = torch::stable::contiguous(encoded_image);
    orientations.push_back(fetch_exif_orientation_from_jpeg_bytes(
        contig.const_data_ptr<uint8_t>(), contig.numel()));
    contig_images.push_back(std::move(contig));
  }

  StableDeviceGuard device_guard(device.index());

  nvjpegOutputFormat_t output_format = output_format_from_mode(mode);

  NVJpegCache& cache = NVJpegCache::get_cache(device);
  std::unique_ptr<CUDAJpegDecoder> decoder = cache.get_decoder(device);

  std::vector<torch::stable::Tensor> result;
  try {
    result = decoder->decode_images(contig_images, output_format);
  } catch (const std::exception& e) {
    // Return the decoder to the pool even on failure so we don't leak it.
    cache.return_decoder(std::move(decoder));
    STD_TORCH_CHECK(false, "Error while decoding JPEG images: ", e.what());
  }
  cache.return_decoder(std::move(decoder));

  // decode_images() host-synchronizes its private stream before returning, so
  // the decoded tensors are fully materialized; applying the EXIF transform
  // (aten flip/transpose on the current stream) is safe. This matches the CPU
  // decoder, which also applies EXIF orientation.
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = exif_orientation_transform(result[i], orientations[i]);
  }
  return result;
}

CUDAJpegDecoder::CUDAJpegDecoder(const torch::stable::Device& target_device)
    : target_device(target_device) {
  StableDeviceGuard device_guard(target_device.index());

  // Pull a stream from torch's pool rather than creating a raw one: a
  // torch-owned stream avoids a cross-DSO teardown hazard, and being a stable
  // private stream (not the caller's ever-changing current stream) means it's
  // safe to cache on this reusable decoder.
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(torch_get_cuda_stream_from_pool(
      /*isHighPriority=*/false, target_device.index(), &stream_ptr));
  stream = static_cast<cudaStream_t>(stream_ptr);

  nvjpegStatus_t status;

  hw_decode_available = true;
  status = nvjpegCreateEx(
      NVJPEG_BACKEND_HARDWARE,
      NULL,
      NULL,
      NVJPEG_FLAGS_DEFAULT,
      &nvjpeg_handle);
  if (status == NVJPEG_STATUS_ARCH_MISMATCH) {
    // No hardware JPEG decoder on this GPU (pre-A100); fall back to the default
    // (software) backend.
    status = nvjpegCreateEx(
        NVJPEG_BACKEND_DEFAULT,
        NULL,
        NULL,
        NVJPEG_FLAGS_DEFAULT,
        &nvjpeg_handle);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize nvjpeg with default backend: ",
        status);
    hw_decode_available = false;
  } else {
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize nvjpeg with hardware backend: ",
        status);
  }

  status = nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg state: ",
      status);

  status = nvjpegDecoderCreate(
      nvjpeg_handle, NVJPEG_BACKEND_DEFAULT, &nvjpeg_decoder);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg decoder: ",
      status);

  status = nvjpegDecoderStateCreate(
      nvjpeg_handle, nvjpeg_decoder, &nvjpeg_decoupled_state);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg decoder state: ",
      status);

  status = nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[0]);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create pinned buffer: ",
      status);

  status = nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[1]);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create pinned buffer: ",
      status);

  status = nvjpegBufferDeviceCreate(nvjpeg_handle, NULL, &device_buffer);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create device buffer: ",
      status);

  status = nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[0]);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create jpeg stream: ",
      status);

  status = nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[1]);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create jpeg stream: ",
      status);

  status = nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create decode params: ",
      status);
}

CUDAJpegDecoder::~CUDAJpegDecoder() {
  // Unlike torchvision (which leaks these to dodge a Windows atexit-vs-CUDA
  // teardown crash), we destroy the nvJPEG handles here. Our decoders are held
  // in NVJpegCache, whose per-device instances are intentionally leaked (never
  // statically destroyed), so this destructor only runs during normal cache
  // eviction while CUDA is alive -- not at process teardown.
  nvjpegDecodeParamsDestroy(nvjpeg_decode_params);
  nvjpegJpegStreamDestroy(jpeg_streams[0]);
  nvjpegJpegStreamDestroy(jpeg_streams[1]);
  nvjpegBufferPinnedDestroy(pinned_buffers[0]);
  nvjpegBufferPinnedDestroy(pinned_buffers[1]);
  nvjpegBufferDeviceDestroy(device_buffer);
  nvjpegJpegStateDestroy(nvjpeg_decoupled_state);
  nvjpegDecoderDestroy(nvjpeg_decoder);
  nvjpegJpegStateDestroy(nvjpeg_state);
  nvjpegDestroy(nvjpeg_handle);
}

std::tuple<
    std::vector<nvjpegImage_t>,
    std::vector<torch::stable::Tensor>,
    std::vector<int>>
CUDAJpegDecoder::prepare_buffers(
    const std::vector<torch::stable::Tensor>& encoded_images,
    const nvjpegOutputFormat_t& output_format) {
  // Scans each JPEG header to size the output tensors and points the
  // nvjpegImage_t channel pointers into that tensor memory.
  int width[NVJPEG_MAX_COMPONENT];
  int height[NVJPEG_MAX_COMPONENT];
  std::vector<int> channels(encoded_images.size());
  nvjpegChromaSubsampling_t subsampling;
  nvjpegStatus_t status;

  std::vector<torch::stable::Tensor> output_tensors;
  output_tensors.reserve(encoded_images.size());
  std::vector<nvjpegImage_t> decoded_images(encoded_images.size());

  for (size_t i = 0; i < encoded_images.size(); ++i) {
    status = nvjpegGetImageInfo(
        nvjpeg_handle,
        encoded_images[i].const_data_ptr<uint8_t>(),
        encoded_images[i].numel(),
        &channels[i],
        &subsampling,
        width,
        height);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS, "Failed to get image info: ", status);
    STD_TORCH_CHECK(
        subsampling != NVJPEG_CSS_UNKNOWN, "Unknown chroma subsampling");

    // Output channels may differ from the source: grayscale is decoded as RGB
    // and sliced back to one channel later (see decode_images).
    int output_channels = (output_format == NVJPEG_OUTPUT_Y) ? 1 : 3;

    auto output_tensor = torch::stable::empty(
        {int64_t(output_channels), int64_t(height[0]), int64_t(width[0])},
        kStableUInt8,
        std::nullopt,
        target_device);

    for (int c = 0; c < output_channels; ++c) {
      decoded_images[i].channel[c] = torch::stable::select(output_tensor, 0, c)
                                         .mutable_data_ptr<uint8_t>();
      decoded_images[i].pitch[c] = width[0];
    }
    for (int c = output_channels; c < NVJPEG_MAX_COMPONENT; ++c) {
      decoded_images[i].channel[c] = NULL;
      decoded_images[i].pitch[c] = 0;
    }
    output_tensors.push_back(output_tensor);
  }
  return {decoded_images, output_tensors, channels};
}

std::vector<torch::stable::Tensor> CUDAJpegDecoder::decode_images(
    const std::vector<torch::stable::Tensor>& encoded_images,
    const nvjpegOutputFormat_t& output_format) {
  // Images are split into two groups: baseline JPEGs (hardware-batch decodable
  // on A100+) and everything else (e.g. progressive), decoded one-by-one in
  // software. See
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/f17940ac4e705bf47a8c39f5365925c1665f6c98/nvJPEG/nvJPEG-Decoder/nvjpegDecoder.cpp#L33
  auto [decoded_imgs_buf, output_tensors, channels] =
      prepare_buffers(encoded_images, output_format);

  nvjpegStatus_t status;
  cudaError_t cudaStatus;

  cudaStatus = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(
      cudaStatus == cudaSuccess,
      "Failed to synchronize CUDA stream: ",
      cudaStatus);

  std::vector<const unsigned char*> hw_input_buffer;
  std::vector<size_t> hw_input_buffer_size;
  std::vector<nvjpegImage_t> hw_output_buffer;

  std::vector<const unsigned char*> sw_input_buffer;
  std::vector<size_t> sw_input_buffer_size;
  std::vector<nvjpegImage_t> sw_output_buffer;

  if (hw_decode_available) {
    for (size_t i = 0; i < encoded_images.size(); ++i) {
      nvjpegJpegStreamParseHeader(
          nvjpeg_handle,
          encoded_images[i].const_data_ptr<uint8_t>(),
          encoded_images[i].numel(),
          jpeg_streams[0]);
      int isSupported = -1;
      nvjpegDecodeBatchedSupported(
          nvjpeg_handle, jpeg_streams[0], &isSupported);

      if (isSupported == 0) {
        hw_input_buffer.push_back(encoded_images[i].const_data_ptr<uint8_t>());
        hw_input_buffer_size.push_back(encoded_images[i].numel());
        hw_output_buffer.push_back(decoded_imgs_buf[i]);
      } else {
        sw_input_buffer.push_back(encoded_images[i].const_data_ptr<uint8_t>());
        sw_input_buffer_size.push_back(encoded_images[i].numel());
        sw_output_buffer.push_back(decoded_imgs_buf[i]);
      }
    }
  } else {
    for (size_t i = 0; i < encoded_images.size(); ++i) {
      sw_input_buffer.push_back(encoded_images[i].const_data_ptr<uint8_t>());
      sw_input_buffer_size.push_back(encoded_images[i].numel());
      sw_output_buffer.push_back(decoded_imgs_buf[i]);
    }
  }

  if (hw_input_buffer.size() > 0) {
    // UNCHANGED is decoded as RGB (see output_format_from_mode).
    status = nvjpegDecodeBatchedInitialize(
        nvjpeg_handle,
        nvjpeg_state,
        hw_input_buffer.size(),
        1,
        output_format == NVJPEG_OUTPUT_UNCHANGED ? NVJPEG_OUTPUT_RGB
                                                 : output_format);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize batch decoding: ",
        status);

    status = nvjpegDecodeBatched(
        nvjpeg_handle,
        nvjpeg_state,
        hw_input_buffer.data(),
        hw_input_buffer_size.data(),
        hw_output_buffer.data(),
        stream);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS, "Failed to decode batch: ", status);
  }

  if (sw_input_buffer.size() > 0) {
    status =
        nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to attach device buffer: ",
        status);
    int buffer_index = 0;
    status = nvjpegDecodeParamsSetOutputFormat(
        nvjpeg_decode_params,
        output_format == NVJPEG_OUTPUT_UNCHANGED ? NVJPEG_OUTPUT_RGB
                                                 : output_format);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to set output format: ",
        status);
    for (size_t i = 0; i < sw_input_buffer.size(); ++i) {
      status = nvjpegJpegStreamParse(
          nvjpeg_handle,
          sw_input_buffer[i],
          sw_input_buffer_size[i],
          0,
          0,
          jpeg_streams[buffer_index]);
      STD_TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to parse jpeg stream: ",
          status);

      status = nvjpegStateAttachPinnedBuffer(
          nvjpeg_decoupled_state, pinned_buffers[buffer_index]);
      STD_TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to attach pinned buffer: ",
          status);

      status = nvjpegDecodeJpegHost(
          nvjpeg_handle,
          nvjpeg_decoder,
          nvjpeg_decoupled_state,
          nvjpeg_decode_params,
          jpeg_streams[buffer_index]);
      STD_TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to decode jpeg stream: ",
          status);

      cudaStatus = cudaStreamSynchronize(stream);
      STD_TORCH_CHECK(
          cudaStatus == cudaSuccess,
          "Failed to synchronize CUDA stream: ",
          cudaStatus);

      status = nvjpegDecodeJpegTransferToDevice(
          nvjpeg_handle,
          nvjpeg_decoder,
          nvjpeg_decoupled_state,
          jpeg_streams[buffer_index],
          stream);
      STD_TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to transfer jpeg to device: ",
          status);

      // Switch pinned buffer to pipeline host and device work.
      buffer_index = 1 - buffer_index;

      status = nvjpegDecodeJpegDevice(
          nvjpeg_handle,
          nvjpeg_decoder,
          nvjpeg_decoupled_state,
          &sw_output_buffer[i],
          stream);
      STD_TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to decode jpeg stream: ",
          status);
    }
  }

  cudaStatus = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(
      cudaStatus == cudaSuccess,
      "Failed to synchronize CUDA stream: ",
      cudaStatus);

  // Prune the extra channels we forced for grayscale-in-UNCHANGED sources.
  if (output_format == NVJPEG_OUTPUT_UNCHANGED) {
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      if (channels[i] == 1) {
        output_tensors[i] = torch::stable::clone(torch::stable::unsqueeze(
            torch::stable::select(output_tensors[i], 0, 0), 0));
      }
    }
  }

  return output_tensors;
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_NVJPEG
