// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include <vector>

#include "StableABICompat.h"

namespace facebook::torchcodec {

FORCE_PUBLIC_VISIBILITY torch::stable::Tensor decode_avif(
    const torch::stable::Tensor& input,
    int64_t mode,
    int64_t output_dtype,
    int64_t num_threads);

FORCE_PUBLIC_VISIBILITY std::vector<int64_t> probe_avif(
    const torch::stable::Tensor& input);

FORCE_PUBLIC_VISIBILITY std::vector<int64_t> probe_avif_from_file(
    std::string filename);

} // namespace facebook::torchcodec
