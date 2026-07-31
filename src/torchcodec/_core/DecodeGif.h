// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "StableABICompat.h"

namespace facebook::torchcodec {

// Decodes a GIF to an (C, H, W) tensor (still GIF) or (N, C, H, W) tensor
// (animated GIF). `mode` is an ImageReadMode: only UNCHANGED, RGB and RGB_ALPHA
// are handled here (the grayscale modes are derived in Python). RGB produces a
// 3-channel tensor with transparency composited over the GIF background color;
// RGB_ALPHA produces a 4-channel RGBA tensor preserving transparency as alpha;
// UNCHANGED produces RGBA if the GIF has any transparency, else RGB.
FORCE_PUBLIC_VISIBILITY torch::stable::Tensor decode_gif(
    const torch::stable::Tensor& input,
    int64_t mode);

} // namespace facebook::torchcodec
