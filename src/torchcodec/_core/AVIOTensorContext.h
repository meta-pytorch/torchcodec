// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "AVIOContextHolder.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

namespace detail {

struct TensorContext {
  torch::stable::Tensor data;
  int64_t current_pos;
  int64_t max_pos;
};

} // namespace detail

// For Decoding: enables users to pass in the entire video or audio as bytes.
// Our read and seek functions then traverse the bytes in memory.
class FORCE_PUBLIC_VISIBILITY AVIOFromTensorContext : public AVIOContextHolder {
 public:
  explicit AVIOFromTensorContext(torch::stable::Tensor data);

  int read(uint8_t* buf, int size) override;
  int64_t seek(int64_t offset, int whence) override;
  int64_t getSize() override;

 private:
  detail::TensorContext tensorContext_;
};

// For Encoding: used to encode into an output uint8 (bytes) tensor.
class FORCE_PUBLIC_VISIBILITY AVIOToTensorContext : public AVIOContextHolder {
 public:
  explicit AVIOToTensorContext();
  torch::stable::Tensor getOutputTensor();

  int write(const uint8_t* buf, int size) override;
  int64_t seek(int64_t offset, int whence) override;
  int64_t getSize() override;

 private:
  detail::TensorContext tensorContext_;
};

} // namespace facebook::torchcodec
