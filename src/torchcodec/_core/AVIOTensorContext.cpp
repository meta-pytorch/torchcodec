// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "AVIOTensorContext.h"

namespace facebook::torchcodec {

namespace {

constexpr int64_t INITIAL_TENSOR_SIZE = 10'000'000; // 10 MB
constexpr int64_t MAX_TENSOR_SIZE = 320'000'000; // 320 MB

// The signature of this function is defined by FFMPEG.
int read(void* opaque, uint8_t* buf, int buf_size) {
  auto tensorContext = static_cast<detail::TensorContext*>(opaque);
  STD_TORCH_CHECK(
      tensorContext->current_pos <= tensorContext->data.numel(),
      "Tried to read outside of the buffer: current_pos=" +
          std::to_string(tensorContext->current_pos) +
          ", size=" + std::to_string(tensorContext->data.numel()));

  int64_t numBytesRead = std::min(
      static_cast<int64_t>(buf_size),
      tensorContext->data.numel() - tensorContext->current_pos);

  STD_TORCH_CHECK(
      numBytesRead >= 0,
      "Tried to read negative bytes: numBytesRead=" +
          std::to_string(numBytesRead) +
          ", size=" + std::to_string(tensorContext->data.numel()) +
          ", current_pos=" + std::to_string(tensorContext->current_pos));

  if (numBytesRead == 0) {
    return AVERROR_EOF;
  }

  std::memcpy(
      buf,
      tensorContext->data.const_data_ptr<uint8_t>() +
          tensorContext->current_pos,
      numBytesRead);
  tensorContext->current_pos += numBytesRead;
  return numBytesRead;
}

// The signature of this function is defined by FFMPEG.
int write(void* opaque, const uint8_t* buf, int buf_size) {
  auto tensorContext = static_cast<detail::TensorContext*>(opaque);

  int64_t bufSize = static_cast<int64_t>(buf_size);
  if (tensorContext->current_pos + bufSize > tensorContext->data.numel()) {
    STD_TORCH_CHECK(
        tensorContext->data.numel() * 2 <= MAX_TENSOR_SIZE,
        "We tried to allocate an output encoded tensor larger than " +
            std::to_string(MAX_TENSOR_SIZE) +
            " bytes. If you think this should be supported, please report.");

    // We double the size of the outpout tensor. Calling stableCat() may not be
    // the most efficient, but it's simple.
    tensorContext->data = stableCat({tensorContext->data, tensorContext->data});
  }

  STD_TORCH_CHECK(
      tensorContext->current_pos + bufSize <= tensorContext->data.numel(),
      "Re-allocation of the output tensor didn't work. "
      "This should not happen, please report on TorchCodec bug tracker");

  uint8_t* outputTensorData = tensorContext->data.mutable_data_ptr<uint8_t>();
  std::memcpy(outputTensorData + tensorContext->current_pos, buf, bufSize);
  tensorContext->current_pos += bufSize;
  // Track the maximum position written so getOutputTensor's narrow() does not
  // truncate the file if final seek was backwards
  tensorContext->max_pos =
      std::max(tensorContext->current_pos, tensorContext->max_pos);
  return buf_size;
}

// The signature of this function is defined by FFMPEG.
int64_t seek(void* opaque, int64_t offset, int whence) {
  auto tensorContext = static_cast<detail::TensorContext*>(opaque);
  int64_t ret = -1;

  switch (whence) {
    case AVSEEK_SIZE:
      ret = tensorContext->data.numel();
      break;
    case SEEK_SET:
      tensorContext->current_pos = offset;
      ret = offset;
      break;
    default:
      break;
  }

  return ret;
}

} // namespace

AVIOFromTensorContext::AVIOFromTensorContext(StableTensor data)
    : tensorContext_{data, 0, 0} {
  STD_TORCH_CHECK(data.numel() > 0, "data must not be empty");
  STD_TORCH_CHECK(stableIsContiguous(data), "data must be contiguous");
  STD_TORCH_CHECK(data.scalar_type() == kStableUInt8, "data must be kUInt8");
  createAVIOContext(
      &read, nullptr, &seek, &tensorContext_, /*isForWriting=*/false);
}

AVIOToTensorContext::AVIOToTensorContext()
    : tensorContext_{
          stableEmpty(
              {INITIAL_TENSOR_SIZE},
              kStableUInt8,
              StableDevice(kStableCPU)),
          0,
          0} {
  createAVIOContext(
      nullptr, &write, &seek, &tensorContext_, /*isForWriting=*/true);
}

StableTensor AVIOToTensorContext::getOutputTensor() {
  return stableNarrow(
      tensorContext_.data,
      /*dim=*/0,
      /*start=*/0,
      /*length=*/tensorContext_.max_pos);
}

} // namespace facebook::torchcodec
