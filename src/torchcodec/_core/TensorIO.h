// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "IOInterface.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

namespace detail {

struct TensorContext {
  torch::stable::Tensor data;
  int64_t current_pos;
  int64_t max_pos;
};

} // namespace detail

// For decoding: reads/seeks over an entire video or audio passed in as a bytes
// tensor. FFmpeg-free.
class FORCE_PUBLIC_VISIBILITY TensorReadIO : public IOInterface {
 public:
  explicit TensorReadIO(torch::stable::Tensor data);

  int read(uint8_t* buf, int size) override;
  int64_t seek(int64_t offset, int whence) override;
  int64_t get_size() override;

 private:
  detail::TensorContext tensor_context_;
};

// For encoding: writes into a growable output uint8 (bytes) tensor.
// FFmpeg-free.
class FORCE_PUBLIC_VISIBILITY TensorWriteIO : public IOInterface {
 public:
  explicit TensorWriteIO();
  torch::stable::Tensor get_output_tensor();

  int write(const uint8_t* buf, int size) override;
  int64_t seek(int64_t offset, int whence) override;
  int64_t get_size() override;

 private:
  detail::TensorContext tensor_context_;
};

} // namespace facebook::torchcodec
