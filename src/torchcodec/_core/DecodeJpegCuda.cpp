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

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <cstring>

#include "Exif.h"
#include "ImageCommon.h"

namespace facebook::torchcodec {

using namespace exif_private;

namespace {

// Scan a full JPEG bitstream for the APP1/EXIF segment and return its
// orientation. The CPU decoder gets EXIF markers from libjpeg's marker list,
// but nvJPEG doesn't expose them, so here we parse the raw bytes ourselves. A
// JPEG is a sequence of marker segments: 0xFFD8 (SOI), then segments of the
// form 0xFF <marker> <2-byte big-endian length> <payload>. The EXIF payload
// lives in an APP1 (0xFFE1) segment and starts with "Exif\0\0".
ExifOrientation fetch_exif_orientation_from_jpeg_bytes(
    const unsigned char* jpeg,
    size_t size) {
  constexpr unsigned char MARKER_PREFIX = 0xFF;
  constexpr unsigned char SOI = 0xD8;
  constexpr unsigned char SOS = 0xDA; // start of scan: no more metadata markers
  constexpr unsigned char EOI = 0xD9;
  constexpr unsigned char APP1 = 0xE1;
  constexpr size_t exif_header_size = 6; // "Exif\0\0"

  if (size < 2 || jpeg[0] != MARKER_PREFIX || jpeg[1] != SOI) {
    return ExifOrientation::Unspecified;
  }

  size_t pos = 2;
  while (pos + 4 <= size && jpeg[pos] == MARKER_PREFIX) {
    unsigned char marker = jpeg[pos + 1];
    if (marker == SOS || marker == EOI) {
      break;
    }
    // Segment length is big-endian and includes the 2 length bytes themselves.
    size_t segment_length =
        (size_t(jpeg[pos + 2]) << 8) | size_t(jpeg[pos + 3]);
    if (segment_length < 2 || pos + 2 + segment_length > size) {
      break;
    }

    if (marker == APP1 && segment_length >= 2 + exif_header_size) {
      const unsigned char* payload = jpeg + pos + 4;
      if (std::memcmp(payload, "Exif\0\0", exif_header_size) == 0) {
        return fetch_exif_orientation(
            payload + exif_header_size, segment_length - 2 - exif_header_size);
      }
    }
    pos += 2 + segment_length;
  }
  return ExifOrientation::Unspecified;
}

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

// nvJPEG decodes UNCHANGED as RGB, so a genuinely grayscale source (1 channel)
// comes out as 3 identical channels; slice it back to a single channel.
torch::stable::Tensor prune_unchanged_grayscale(
    const torch::stable::Tensor& output_tensor,
    int source_channels,
    const nvjpegOutputFormat_t& output_format) {
  if (output_format == NVJPEG_OUTPUT_UNCHANGED && source_channels == 1) {
    return torch::stable::clone(torch::stable::unsqueeze(
        torch::stable::select(output_tensor, 0, 0), 0));
  }
  return output_tensor;
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

  int device_index = resolve_device_index(device);
  StableDeviceGuard device_guard(device_index);

  // Decode on the caller's current stream (honors torch.cuda.Stream()), fetched
  // via the stable-ABI shim so the FFmpeg-free image lib needs no FFmpeg
  // headers.
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  nvjpegOutputFormat_t output_format = output_format_from_mode(mode);

  NVJpegCache& cache = NVJpegCache::get_cache(device);
  std::unique_ptr<CUDAJpegDecoder> decoder = cache.get_decoder(device);

  std::vector<torch::stable::Tensor> result;
  try {
    result = decoder->decode_images(contig_images, output_format, stream);
  } catch (const std::exception& e) {
    // Return the decoder to the pool even on failure so we don't leak it.
    cache.return_decoder(std::move(decoder));
    STD_TORCH_CHECK(false, "Error while decoding JPEG images: ", e.what());
  }
  cache.return_decoder(std::move(decoder));

  // decode_images() host-synchronizes the decode stream before returning, so
  // the decoded tensors are fully materialized; applying the EXIF transform
  // (aten flip/transpose on the current stream) is safe. This matches the CPU
  // decoder, which also applies EXIF orientation.
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = exif_orientation_transform(result[i], orientations[i]);
  }
  return result;
}

CUDAJpegDecoder::CUDAJpegDecoder(const torch::stable::Device& target_device)
    : target_device_(target_device) {
  StableDeviceGuard device_guard(target_device_.index());

  nvjpegStatus_t status;

  hw_decode_available_ = true;
  status = nvjpegCreateEx(
      NVJPEG_BACKEND_HARDWARE,
      NULL,
      NULL,
      NVJPEG_FLAGS_DEFAULT,
      &nvjpeg_handle_);
  if (status == NVJPEG_STATUS_ARCH_MISMATCH) {
    // No hardware JPEG decoder on this GPU (pre-A100); fall back to the default
    // (software) backend.
    status = nvjpegCreateEx(
        NVJPEG_BACKEND_DEFAULT,
        NULL,
        NULL,
        NVJPEG_FLAGS_DEFAULT,
        &nvjpeg_handle_);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize nvjpeg with default backend: ",
        status);
    hw_decode_available_ = false;
  } else {
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize nvjpeg with hardware backend: ",
        status);
  }

  status = nvjpegJpegStateCreate(nvjpeg_handle_, &nvjpeg_state_);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg state: ",
      status);

  status = nvjpegDecoderCreate(
      nvjpeg_handle_, NVJPEG_BACKEND_DEFAULT, &nvjpeg_decoder_);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg decoder: ",
      status);

  status = nvjpegDecoderStateCreate(
      nvjpeg_handle_, nvjpeg_decoder_, &nvjpeg_decoupled_state_);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg decoder state: ",
      status);

  status = nvjpegBufferPinnedCreate(nvjpeg_handle_, NULL, &pinned_buffers_[0]);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create pinned buffer: ",
      status);

  status = nvjpegBufferPinnedCreate(nvjpeg_handle_, NULL, &pinned_buffers_[1]);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create pinned buffer: ",
      status);

  status = nvjpegBufferDeviceCreate(nvjpeg_handle_, NULL, &device_buffer_);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create device buffer: ",
      status);

  status = nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_streams_[0]);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create jpeg stream: ",
      status);

  status = nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_streams_[1]);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create jpeg stream: ",
      status);

  status = nvjpegDecodeParamsCreate(nvjpeg_handle_, &nvjpeg_decode_params_);
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
  nvjpegDecodeParamsDestroy(nvjpeg_decode_params_);
  nvjpegJpegStreamDestroy(jpeg_streams_[0]);
  nvjpegJpegStreamDestroy(jpeg_streams_[1]);
  nvjpegBufferPinnedDestroy(pinned_buffers_[0]);
  nvjpegBufferPinnedDestroy(pinned_buffers_[1]);
  nvjpegBufferDeviceDestroy(device_buffer_);
  nvjpegJpegStateDestroy(nvjpeg_decoupled_state_);
  nvjpegDecoderDestroy(nvjpeg_decoder_);
  nvjpegJpegStateDestroy(nvjpeg_state_);
  nvjpegDestroy(nvjpeg_handle_);
}

std::tuple<torch::stable::Tensor, nvjpegImage_t, int>
CUDAJpegDecoder::allocate_output_image(
    const torch::stable::Tensor& encoded_image,
    const nvjpegOutputFormat_t& output_format) {
  // Scan the JPEG header to size the output tensor, allocate it, and point an
  // nvjpegImage_t's channel pointers into that tensor's memory. Also returns
  // the number of channels in the source (for UNCHANGED grayscale pruning).
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int source_channels = 0;
  nvjpegChromaSubsampling_t subsampling;
  nvjpegStatus_t status = nvjpegGetImageInfo(
      nvjpeg_handle_,
      encoded_image.const_data_ptr<uint8_t>(),
      encoded_image.numel(),
      &source_channels,
      &subsampling,
      widths,
      heights);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS, "Failed to get image info: ", status);
  STD_TORCH_CHECK(
      subsampling != NVJPEG_CSS_UNKNOWN, "Unknown chroma subsampling");

  // Output channels may differ from the source: grayscale is decoded as RGB and
  // sliced back to one channel later (see prune_unchanged_grayscale).
  int output_channels = (output_format == NVJPEG_OUTPUT_Y) ? 1 : 3;

  auto output_tensor = torch::stable::empty(
      {int64_t(output_channels), int64_t(heights[0]), int64_t(widths[0])},
      kStableUInt8,
      std::nullopt,
      target_device_);

  nvjpegImage_t nvjpeg_image;
  for (int c = 0; c < output_channels; ++c) {
    nvjpeg_image.channel[c] =
        torch::stable::select(output_tensor, 0, c).mutable_data_ptr<uint8_t>();
    nvjpeg_image.pitch[c] = widths[0];
  }
  for (int c = output_channels; c < NVJPEG_MAX_COMPONENT; ++c) {
    nvjpeg_image.channel[c] = nullptr;
    nvjpeg_image.pitch[c] = 0;
  }
  return {output_tensor, nvjpeg_image, source_channels};
}

bool CUDAJpegDecoder::is_hw_batched_supported(
    const unsigned char* data,
    size_t size) {
  nvjpegJpegStreamParseHeader(nvjpeg_handle_, data, size, jpeg_streams_[0]);
  int is_supported = -1;
  nvjpegDecodeBatchedSupported(nvjpeg_handle_, jpeg_streams_[0], &is_supported);
  return is_supported == 0; // nvJPEG sets 0 when supported
}

std::pair<std::vector<size_t>, std::vector<size_t>>
CUDAJpegDecoder::split_images_by_backend(
    const std::vector<torch::stable::Tensor>& encoded_images) {
  // Baseline JPEGs (hardware-batch decodable on A100+) go to the hardware
  // group; everything else (e.g. progressive), and all images when there's no
  // HW engine, go to the software group. See
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/f17940ac4e705bf47a8c39f5365925c1665f6c98/nvJPEG/nvJPEG-Decoder/nvjpegDecoder.cpp#L33
  std::vector<size_t> hw_indices, sw_indices;
  for (size_t i = 0; i < encoded_images.size(); ++i) {
    bool supports_hw = hw_decode_available_ &&
        is_hw_batched_supported(
                           encoded_images[i].const_data_ptr<uint8_t>(),
                           encoded_images[i].numel());
    (supports_hw ? hw_indices : sw_indices).push_back(i);
  }
  return {std::move(hw_indices), std::move(sw_indices)};
}

void CUDAJpegDecoder::decode_batched_hardware(
    const std::vector<torch::stable::Tensor>& encoded_images,
    const std::vector<size_t>& indices,
    const nvjpegOutputFormat_t& output_format,
    cudaStream_t stream,
    std::vector<torch::stable::Tensor>& output_tensors) {
  // The batched API needs the batch as contiguous parallel arrays: input
  // pointers, sizes, and output nvjpegImage_t. Allocate the outputs here and
  // gather everything for this group's (scattered) indices.
  std::vector<const unsigned char*> inputs;
  std::vector<size_t> sizes;
  std::vector<nvjpegImage_t> nvjpeg_images;
  std::vector<int> source_channels;
  inputs.reserve(indices.size());
  sizes.reserve(indices.size());
  nvjpeg_images.reserve(indices.size());
  source_channels.reserve(indices.size());
  for (size_t idx : indices) {
    auto [output_tensor, nvjpeg_image, channels] =
        allocate_output_image(encoded_images[idx], output_format);
    output_tensors[idx] = output_tensor;
    inputs.push_back(encoded_images[idx].const_data_ptr<uint8_t>());
    sizes.push_back(encoded_images[idx].numel());
    nvjpeg_images.push_back(nvjpeg_image);
    source_channels.push_back(channels);
  }

  // We use the batched API even for a single image (batch size 1) on purpose:
  // it's the only entry point to nvJPEG's dedicated hardware JPEG engine, and
  // this path only runs when that engine is available (hw_decode_available_,
  // A100+). The simple single-image nvjpegDecode() uses the default/hybrid GPU
  // backend, not the HW engine, so it wouldn't be faster here.
  //
  // UNCHANGED is decoded as RGB (see output_format_from_mode).
  nvjpegStatus_t status = nvjpegDecodeBatchedInitialize(
      nvjpeg_handle_,
      nvjpeg_state_,
      nvjpeg_images.size(),
      1,
      output_format == NVJPEG_OUTPUT_UNCHANGED ? NVJPEG_OUTPUT_RGB
                                               : output_format);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to initialize batch decoding: ",
      status);

  status = nvjpegDecodeBatched(
      nvjpeg_handle_,
      nvjpeg_state_,
      inputs.data(),
      sizes.data(),
      nvjpeg_images.data(),
      stream);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS, "Failed to decode batch: ", status);

  for (size_t i = 0; i < indices.size(); ++i) {
    output_tensors[indices[i]] = prune_unchanged_grayscale(
        output_tensors[indices[i]], source_channels[i], output_format);
  }
}

void CUDAJpegDecoder::decode_software(
    const std::vector<torch::stable::Tensor>& encoded_images,
    const std::vector<size_t>& indices,
    const nvjpegOutputFormat_t& output_format,
    cudaStream_t stream,
    std::vector<torch::stable::Tensor>& output_tensors) {
  // Decoupled host/device pipeline: Huffman decode runs on the CPU
  // (DecodeJpegHost), the coefficients are copied to the device
  // (TransferToDevice), and IDCT/upsampling/color conversion run on the GPU
  // (DecodeJpegDevice). Two pinned buffers are ping-ponged so image i+1's host
  // work overlaps image i's device work.
  nvjpegStatus_t status =
      nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state_, device_buffer_);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to attach device buffer: ",
      status);

  // UNCHANGED is decoded as RGB (see output_format_from_mode).
  status = nvjpegDecodeParamsSetOutputFormat(
      nvjpeg_decode_params_,
      output_format == NVJPEG_OUTPUT_UNCHANGED ? NVJPEG_OUTPUT_RGB
                                               : output_format);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS, "Failed to set output format: ", status);

  int buffer_index = 0;
  for (size_t idx : indices) {
    auto [output_tensor, nvjpeg_image, source_channels] =
        allocate_output_image(encoded_images[idx], output_format);

    status = nvjpegJpegStreamParse(
        nvjpeg_handle_,
        encoded_images[idx].const_data_ptr<uint8_t>(),
        encoded_images[idx].numel(),
        0,
        0,
        jpeg_streams_[buffer_index]);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to parse jpeg stream: ",
        status);

    status = nvjpegStateAttachPinnedBuffer(
        nvjpeg_decoupled_state_, pinned_buffers_[buffer_index]);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to attach pinned buffer: ",
        status);

    status = nvjpegDecodeJpegHost(
        nvjpeg_handle_,
        nvjpeg_decoder_,
        nvjpeg_decoupled_state_,
        nvjpeg_decode_params_,
        jpeg_streams_[buffer_index]);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to decode jpeg stream: ",
        status);

    cudaError_t cuda_status = cudaStreamSynchronize(stream);
    STD_TORCH_CHECK(
        cuda_status == cudaSuccess,
        "Failed to synchronize CUDA stream: ",
        cuda_status);

    status = nvjpegDecodeJpegTransferToDevice(
        nvjpeg_handle_,
        nvjpeg_decoder_,
        nvjpeg_decoupled_state_,
        jpeg_streams_[buffer_index],
        stream);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to transfer jpeg to device: ",
        status);

    // Switch pinned buffer to pipeline host and device work.
    buffer_index = 1 - buffer_index;

    status = nvjpegDecodeJpegDevice(
        nvjpeg_handle_,
        nvjpeg_decoder_,
        nvjpeg_decoupled_state_,
        &nvjpeg_image,
        stream);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to decode jpeg stream: ",
        status);

    output_tensors[idx] = prune_unchanged_grayscale(
        output_tensor, source_channels, output_format);
  }
}

std::vector<torch::stable::Tensor> CUDAJpegDecoder::decode_images(
    const std::vector<torch::stable::Tensor>& encoded_images,
    const nvjpegOutputFormat_t& output_format,
    cudaStream_t stream) {
  std::vector<torch::stable::Tensor> output_tensors(encoded_images.size());

  auto [hw_indices, sw_indices] = split_images_by_backend(encoded_images);
  if (!hw_indices.empty()) {
    decode_batched_hardware(
        encoded_images, hw_indices, output_format, stream, output_tensors);
  }
  if (!sw_indices.empty()) {
    decode_software(
        encoded_images, sw_indices, output_format, stream, output_tensors);
  }

  // Host-synchronize before returning: the decoder (and its internal nvJPEG
  // buffers) goes back to the pool and may be reused immediately by the next
  // call, so all GPU work using them must complete first.
  cudaError_t cuda_status = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(
      cuda_status == cudaSuccess,
      "Failed to synchronize CUDA stream: ",
      cuda_status);

  return output_tensors;
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_NVJPEG
