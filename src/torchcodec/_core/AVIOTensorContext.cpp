// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "AVIOTensorContext.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

namespace {

constexpr int64_t INITIAL_TENSOR_SIZE = 10'000'000; // 10 MB
constexpr int64_t MAX_TENSOR_SIZE = 320'000'000; // 320 MB

// The signature of this function is defined by FFMPEG.
int readCallback(void* opaque, uint8_t* buf, int buf_size) {
  auto self = static_cast<AVIOFromTensorContext*>(opaque);
  int result = self->read(buf, buf_size);
  return result < 0 ? AVERROR_EOF : result;
}

// The signature of this function is defined by FFMPEG.
int write(void* opaque, const uint8_t* buf, int buf_size) {
  auto tensorContext = static_cast<detail::TensorContext*>(opaque);

  int64_t bufSize = static_cast<int64_t>(buf_size);
  if (tensorContext->current_pos + bufSize > tensorContext->data.numel()) {
    STD_TORCH_CHECK(
        tensorContext->data.numel() * 2 <= MAX_TENSOR_SIZE,
        "We tried to allocate an output encoded tensor larger than ",
        MAX_TENSOR_SIZE,
        " bytes. If you think this should be supported, please report.");

    // We double the size of the outpout tensor. Calling cat() may not be the
    // most efficient, but it's simple.
    tensorContext->data =
        stableCat({tensorContext->data, tensorContext->data}, 0);
  }

  STD_TORCH_CHECK(
      tensorContext->current_pos + bufSize <= tensorContext->data.numel(),
      "Re-allocation of the output tensor didn't work. ",
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
int64_t seekCallback(void* opaque, int64_t offset, int whence) {
  auto self = static_cast<AVIOFromTensorContext*>(opaque);
  if (whence == AVSEEK_SIZE) {
    return self->getSize();
  }
  return self->seek(offset, whence);
}

// The signature of this function is defined by FFMPEG.
int64_t seekWrite(void* opaque, int64_t offset, int whence) {
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

AVIOFromTensorContext::AVIOFromTensorContext(torch::stable::Tensor data)
    : tensorContext_{data, 0, 0} {
  STD_TORCH_CHECK(data.numel() > 0, "data must not be empty");
  STD_TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
  STD_TORCH_CHECK(data.scalar_type() == kStableUInt8, "data must be kUInt8");
  createAVIOContext(
      &readCallback, nullptr, &seekCallback, this, /*isForWriting=*/false);
}

int AVIOFromTensorContext::read(uint8_t* buf, int size) {
  if (tensorContext_.current_pos >= tensorContext_.data.numel()) {
    return -1;
  }

  int64_t numBytesRead = std::min(
      static_cast<int64_t>(size),
      tensorContext_.data.numel() - tensorContext_.current_pos);

  STD_TORCH_CHECK(
      numBytesRead >= 0,
      "Tried to read negative bytes: numBytesRead=",
      numBytesRead,
      ", size=",
      tensorContext_.data.numel(),
      ", current_pos=",
      tensorContext_.current_pos);

  if (numBytesRead == 0) {
    return -1;
  }

  std::memcpy(
      buf,
      tensorContext_.data.const_data_ptr<uint8_t>() +
          tensorContext_.current_pos,
      numBytesRead);
  tensorContext_.current_pos += numBytesRead;
  return static_cast<int>(numBytesRead);
}

int64_t AVIOFromTensorContext::seek(int64_t offset, int whence) {
  switch (whence) {
    case SEEK_SET:
      tensorContext_.current_pos = offset;
      return offset;
    default:
      return -1;
  }
}

int64_t AVIOFromTensorContext::getSize() {
  return tensorContext_.data.numel();
}

AVIOToTensorContext::AVIOToTensorContext()
    : tensorContext_{
          torch::stable::empty({INITIAL_TENSOR_SIZE}, kStableUInt8),
          0,
          0} {
  createAVIOContext(
      nullptr, &write, &seekWrite, &tensorContext_, /*isForWriting=*/true);
}

torch::stable::Tensor AVIOToTensorContext::getOutputTensor() {
  return torch::stable::narrow(
      tensorContext_.data,
      /*dim=*/0,
      /*start=*/0,
      /*length=*/tensorContext_.max_pos);
}

} // namespace facebook::torchcodec
