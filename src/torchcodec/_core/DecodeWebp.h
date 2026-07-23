// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "StableABICompat.h"

namespace facebook::torchcodec {

FORCE_PUBLIC_VISIBILITY torch::stable::Tensor decode_webp(
    const torch::stable::Tensor& input,
    int64_t mode);

} // namespace facebook::torchcodec
