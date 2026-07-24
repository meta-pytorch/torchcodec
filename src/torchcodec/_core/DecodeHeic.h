// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "StableABICompat.h"

namespace facebook::torchcodec {

// Decodes an HEIC/HEIF image to a (C, H, W) tensor at the source's NATIVE
// precision: uint8 for 8-bit sources, uint16 (mapped to the full range) for
// >8-bit (10/12-bit) sources. `mode` is an ImageReadMode: only UNCHANGED, RGB
// and RGB_ALPHA are handled here (the grayscale modes are derived in Python).
// The output_dtype conversion (forcing uint8 or uint16) is likewise applied in
// Python (see decode_heic in _image_decoders.py), because libheif's chroma
// enum couples the decode bit-depth to the output, so it's simpler and safer to
// decode natively here and convert afterwards.
//
// Unlike the other image decoders, this lives in its own separately-loadable
// libtorchcodec_heic library that links libheif (LGPL). libheif is NOT bundled
// in our wheels: it's an optional, user-supplied runtime dependency (mirroring
// how we treat FFmpeg), so the library is loaded lazily on first use.
FORCE_PUBLIC_VISIBILITY torch::stable::Tensor decode_heic(
    const torch::stable::Tensor& input,
    int64_t mode);

} // namespace facebook::torchcodec
