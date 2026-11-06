// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>

#include "StableABICompat.h"

namespace facebook::torchcodec {

FORCE_PUBLIC_VISIBILITY torch::stable::Tensor decode_png(
    const torch::stable::Tensor& input,
    int64_t mode,
    int64_t output_dtype);

// Returns the offset of the first chunk after the run of consecutive IDAT
// chunks starting at `offset`, or an offset at or past the end of the input if
// the run reaches it. get_image_metadata() needs the chunks after the image
// data, and walking over the (often thousands of) IDAT chunks in Python is
// slow. These don't need libpng.
FORCE_PUBLIC_VISIBILITY int64_t png_skip_idat_chunks_from_tensor(
    const torch::stable::Tensor& input,
    int64_t offset);
FORCE_PUBLIC_VISIBILITY int64_t
png_skip_idat_chunks_from_file(std::string filename, int64_t offset);

} // namespace facebook::torchcodec
