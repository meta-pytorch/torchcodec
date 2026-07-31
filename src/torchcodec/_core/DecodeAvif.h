// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "StableABICompat.h"

namespace facebook::torchcodec {

FORCE_PUBLIC_VISIBILITY torch::stable::Tensor decode_avif(
    const torch::stable::Tensor& input,
    int64_t mode,
    int64_t output_dtype,
    int64_t num_threads);

} // namespace facebook::torchcodec
