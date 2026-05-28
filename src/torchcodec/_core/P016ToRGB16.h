// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace facebook::torchcodec {

// Launch a CUDA kernel that converts a P016 (16-bit semi-planar YUV 4:2:0)
// frame to interleaved RGB16 (uint16 per channel, HWC layout).
//
// The colorMatrix is a 3x4 row-major matrix that operates on raw P016 values
// (after right-shifting to bitDepth precision). Each row produces one RGB
// channel: out = clamp(m[0]*Y + m[1]*U + m[2]*V + m[3], 0, 65535)
void launchP016ToRGB16Kernel(
    const uint16_t* yPlane,
    const uint16_t* uvPlane,
    uint16_t* rgbOutput,
    int width,
    int height,
    int yPitch,
    int uvPitch,
    int rgbPitch,
    int bitDepth,
    const float colorMatrix[3][4],
    cudaStream_t stream);

} // namespace facebook::torchcodec
