// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "StableABICompat.h"

namespace facebook::torchcodec {

// GIF decoding always produces RGB output, so unlike the other image decoders
// there is no `mode` parameter here: color-mode conversion is done in Python.
// The output is (C, H, W) for a single-image GIF and (N, C, H, W) for an
// animated one.
FORCE_PUBLIC_VISIBILITY torch::stable::Tensor decode_gif(
    const torch::stable::Tensor& data);

} // namespace facebook::torchcodec
