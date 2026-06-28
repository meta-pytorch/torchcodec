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
} // namespace

// --------------------------------------------------------------------------
// AVIOFromTensorContext
// --------------------------------------------------------------------------

AVIOFromTensorContext::AVIOFromTensorContext(torch::stable::Tensor data)
    : tensor_context_{data, 0, 0} {
  STD_TORCH_CHECK(data.numel() > 0, "data must not be empty");
  STD_TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
  STD_TORCH_CHECK(data.scalar_type() == kStableUInt8, "data must be kUInt8");
  create_avio_context(/*isForWriting=*/false);
}

int AVIOFromTensorContext::read(uint8_t* buf, int size) {
  if (tensor_context_.current_pos >= tensor_context_.data.numel()) {
    return -1;
  }

  int64_t num_bytes_read = std::min(
      static_cast<int64_t>(size),
      tensor_context_.data.numel() - tensor_context_.current_pos);

  STD_TORCH_CHECK(
      num_bytes_read >= 0,
      "Tried to read negative bytes: numBytesRead=",
      num_bytes_read,
      ", size=",
      tensor_context_.data.numel(),
      ", current_pos=",
      tensor_context_.current_pos);

  if (num_bytes_read == 0) {
    return -1;
  }

  std::memcpy(
      buf,
      tensor_context_.data.const_data_ptr<uint8_t>() +
          tensor_context_.current_pos,
      num_bytes_read);
  tensor_context_.current_pos += num_bytes_read;
  return static_cast<int>(num_bytes_read);
}

int64_t AVIOFromTensorContext::seek(int64_t offset, int whence) {
  switch (whence) {
    case SEEK_SET:
      tensor_context_.current_pos = offset;
      return offset;
    default:
      return -1;
  }
}

int64_t AVIOFromTensorContext::get_size() {
  return tensor_context_.data.numel();
}

// --------------------------------------------------------------------------
// AVIOToTensorContext
// --------------------------------------------------------------------------

AVIOToTensorContext::AVIOToTensorContext()
    : tensor_context_{
          torch::stable::empty({INITIAL_TENSOR_SIZE}, kStableUInt8),
          0,
          0} {
  create_avio_context(/*isForWriting=*/true);
}

int AVIOToTensorContext::write(const uint8_t* buf, int size) {
  int64_t buf_size = static_cast<int64_t>(size);
  if (tensor_context_.current_pos + buf_size > tensor_context_.data.numel()) {
    STD_TORCH_CHECK(
        tensor_context_.data.numel() * 2 <= MAX_TENSOR_SIZE,
        "We tried to allocate an output encoded tensor larger than ",
        MAX_TENSOR_SIZE,
        " bytes. If you think this should be supported, please report.");

    // We double the size of the outpout tensor. Calling cat() may not be the
    // most efficient, but it's simple.
    tensor_context_.data =
        stable_cat({tensor_context_.data, tensor_context_.data}, 0);
  }

  STD_TORCH_CHECK(
      tensor_context_.current_pos + buf_size <= tensor_context_.data.numel(),
      "Re-allocation of the output tensor didn't work. ",
      "This should not happen, please report on TorchCodec bug tracker");

  uint8_t* output_tensor_data =
      tensor_context_.data.mutable_data_ptr<uint8_t>();
  std::memcpy(output_tensor_data + tensor_context_.current_pos, buf, buf_size);
  tensor_context_.current_pos += buf_size;
  // Track the maximum position written so getOutputTensor's narrow() does not
  // truncate the file if final seek was backwards
  tensor_context_.max_pos =
      std::max(tensor_context_.current_pos, tensor_context_.max_pos);
  return size;
}

int64_t AVIOToTensorContext::seek(int64_t offset, int whence) {
  switch (whence) {
    case SEEK_SET:
      tensor_context_.current_pos = offset;
      return offset;
    default:
      return -1;
  }
}

int64_t AVIOToTensorContext::get_size() {
  return tensor_context_.data.numel();
}

torch::stable::Tensor AVIOToTensorContext::get_output_tensor() {
  return torch::stable::narrow(
      tensor_context_.data,
      /*dim=*/0,
      /*start=*/0,
      /*length=*/tensor_context_.max_pos);
}

} // namespace facebook::torchcodec
