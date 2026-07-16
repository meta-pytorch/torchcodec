// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "StableABICompat.h"

namespace facebook::torchcodec {

// CPU JPEG decoder. Decodes the encoded bytes in `data` (a contiguous 1-D uint8
// tensor) into a (C, H, W) uint8 tensor. `mode` matches the Python
// ImageColorMode enum. EXIF orientation is always applied. If torchcodec was
// built without libjpeg, this raises an actionable error.
torch::stable::Tensor decode_jpeg(
    const torch::stable::Tensor& data,
    int64_t mode);

} // namespace facebook::torchcodec
