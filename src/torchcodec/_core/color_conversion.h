// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <optional>

#include "FFMPEGCommon.h"
#include "Frame.h"

namespace facebook::torchcodec {

struct LumaCoefficients {
  float kr, kg, kb;
};

LumaCoefficients getLumaCoefficients(AVColorSpace colorspace);

struct CachedColorMatrix {
  AVColorSpace colorspace = AVCOL_SPC_UNSPECIFIED;
  AVColorRange colorRange = AVCOL_RANGE_UNSPECIFIED;
  int bitDepth = 0;
  float outScale = 0;
  float matrix[3][4] = {};
  bool valid = false;
};

// Compute the YUV -> RGB color conversion matrix.
//
// The matrix operates on raw YUV sample values (after any bit-shifting for
// P016) and produces RGB values in [0, outScale]. The 4th column encodes
// all additive offsets including UV centering.
//
// bitDepth: 8 for NV12, 10 or 12 for P016.
// outScale: 255.0 for NV12 (uint8 output), 65535.0 for P016 (uint16 output).
void computeColorConversionMatrix(
    AVColorSpace colorspace,
    AVColorRange colorRange,
    int bitDepth,
    float outScale,
    float outMatrix[3][4]);

void maybeUpdateColorMatrix(
    CachedColorMatrix& cachedColorMatrix,
    AVColorSpace colorspace,
    AVColorRange colorRange,
    int bitDepth,
    float outScale);

void launchNV12ToRGBKernel(
    const uint8_t* yPlane,
    const uint8_t* uvPlane,
    uint8_t* rgbOutput,
    int width,
    int height,
    int yPitch,
    int uvPitch,
    int rgbPitch,
    const float colorMatrix[3][4],
    cudaStream_t stream);

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

// Convert a YUV frame (NV12 or P016) on GPU to an interleaved RGB tensor.
//
// isP016: true for P016 (uint16 I/O), false for NV12 (uint8 I/O).
// bitDepth: 8 for NV12, actual bit depth (10 or 12) for P016.
// outputDims: desired output size; if the frame was rounded up to even
//   dimensions, the result is cropped back to outputDims.
torch::stable::Tensor convertYUVFrameToRGB(
    UniqueAVFrame& avFrame,
    const StableDevice& device,
    cudaStream_t nvdecStream,
    std::optional<torch::stable::Tensor> preAllocatedOutputTensor,
    const FrameDims& outputDims,
    bool isP016,
    int bitDepth,
    CachedColorMatrix& cachedColorMatrix);

// Compute the RGB -> YUV color conversion matrix (for encoding).
// The matrix operates on uint8 RGB values [0, 255] and produces Y, U, V
// values. The 4th column encodes offsets (Y pedestal for limited range,
// UV centering at 128).
void computeRGBToYUVMatrix(
    AVColorSpace colorspace,
    AVColorRange colorRange,
    float outMatrix[3][4]);

void launchRGBToNV12Kernel(
    const uint8_t* rgbInput,
    uint8_t* yPlane,
    uint8_t* uvPlane,
    int width,
    int height,
    int rgbPitch,
    int yPitch,
    int uvPitch,
    const float colorMatrix[3][4],
    cudaStream_t stream);

} // namespace facebook::torchcodec
