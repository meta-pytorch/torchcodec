// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "EncodePng.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !TORCHCODEC_ENABLE_PNG

namespace facebook::torchcodec {

torch::stable::Tensor encode_png(
    [[maybe_unused]] const torch::stable::Tensor& data,
    [[maybe_unused]] int64_t compression_level) {
  STD_TORCH_CHECK(
      false,
      "encode_png: torchcodec was not compiled with libpng support. "
      "Rebuild torchcodec in an environment where libpng (and its development "
      "headers) are available. If you see this error in a prebuilt wheel, "
      "please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

namespace facebook::torchcodec {

torch::stable::Tensor encode_png(
    [[maybe_unused]] const torch::stable::Tensor& data,
    [[maybe_unused]] int64_t compression_level) {
  STD_TORCH_CHECK(false, "encode_png: not yet implemented.");
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_PNG
