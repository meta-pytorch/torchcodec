// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "IOInterface.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

// Encodes a CHW uint8 image tensor as PNG, streaming the bytes directly into
// the given IOInterface (a file on disk or a Python file-like object).
FORCE_PUBLIC_VISIBILITY void encode_png_to_io(
    const torch::stable::Tensor& img,
    int64_t compression_level,
    IOInterface& io);

} // namespace facebook::torchcodec
