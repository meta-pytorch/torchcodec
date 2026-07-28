// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "EncodeJpeg.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !TORCHCODEC_ENABLE_JPEG

namespace facebook::torchcodec {

torch::stable::Tensor encode_jpeg(
    [[maybe_unused]] const torch::stable::Tensor& img,
    [[maybe_unused]] int64_t quality) {
  STD_TORCH_CHECK(
      false,
      "encode_jpeg: torchcodec was not compiled with libjpeg support. "
      "Rebuild torchcodec in an environment where libjpeg-turbo (and its "
      "development headers) are available. If you see this error in a prebuilt "
      "wheel, please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

namespace facebook::torchcodec {

torch::stable::Tensor encode_jpeg(
    [[maybe_unused]] const torch::stable::Tensor& img,
    [[maybe_unused]] int64_t quality) {
  STD_TORCH_CHECK(false, "encode_jpeg: not yet implemented.");
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_JPEG
