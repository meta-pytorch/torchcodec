// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "P016ToRGB16.h"

namespace facebook::torchcodec {

// P016 is the 16bit equivalent of NV12. Chroma subsampling is 4:2:0 so U,V are
// subsampled by 2 in both H and W dimensions.
// The Y plane is YYYYY... where each Y is 16bits and corresponds to a single
// pixel
// The UV plane is interleaved as UVUVUVUV... where each U and each V is 16bits,
// and each UV pair corresponds to a 2x2 block of pixels.
//
// The actual data is stored in the most significant bits of each 16-bit word,
// with the lower bits zero-padded (e.g. 10-bit content occupies the upper 10
// bits).


// Color conversion matrix stored in constant memory for fast access.
// It'll be available to every single thread (each running p016ToRgb16Kernel
// individually).
__constant__ float d_colorMatrix[3][4];

// Takes the Y and UV plane as input, applies the color-conversion matrix and
// fills the RGB plane as output.
__global__ void p016ToRgb16Kernel(
    // __restrict__ tells the compiler those pointers never overlap with each
    // other so it can optimize read and writes more aggressively.
    const uint16_t* __restrict__ yPlane,
    const uint16_t* __restrict__ uvPlane,
    uint16_t* __restrict__ rgbOutput,
    int width,
    int height,
    int yPitchElements,
    int uvPitchElements,
    int rgbPitchElements,
    int bitShift) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  // Read Y value and shift to actual bit depth
  float yVal =
      static_cast<float>(yPlane[y * yPitchElements + x] >> bitShift);

  // Read U, V values (4:2:0 chroma subsampling: UV is half resolution)
  int uvX = x / 2;
  int uvY = y / 2;
  int uvIdx = uvY * uvPitchElements + uvX * 2;
  float uVal = static_cast<float>(uvPlane[uvIdx] >> bitShift);
  float vVal = static_cast<float>(uvPlane[uvIdx + 1] >> bitShift);

  // Apply 3x4 color conversion matrix:
  //   R = m[0][0]*Y + m[0][1]*U + m[0][2]*V + m[0][3]
  //   G = m[1][0]*Y + m[1][1]*U + m[1][2]*V + m[1][3]
  //   B = m[2][0]*Y + m[2][1]*U + m[2][2]*V + m[2][3]
  float r = d_colorMatrix[0][0] * yVal + d_colorMatrix[0][1] * uVal +
      d_colorMatrix[0][2] * vVal + d_colorMatrix[0][3];
  float g = d_colorMatrix[1][0] * yVal + d_colorMatrix[1][1] * uVal +
      d_colorMatrix[1][2] * vVal + d_colorMatrix[1][3];
  float b = d_colorMatrix[2][0] * yVal + d_colorMatrix[2][1] * uVal +
      d_colorMatrix[2][2] * vVal + d_colorMatrix[2][3];

  // Clamp to [0, 65535] and write RGB output
  int rgbIdx = y * rgbPitchElements + x * 3;
  rgbOutput[rgbIdx + 0] =
      static_cast<uint16_t>(fminf(fmaxf(r, 0.0f), 65535.0f));
  rgbOutput[rgbIdx + 1] =
      static_cast<uint16_t>(fminf(fmaxf(g, 0.0f), 65535.0f));
  rgbOutput[rgbIdx + 2] =
      static_cast<uint16_t>(fminf(fmaxf(b, 0.0f), 65535.0f));
}

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
    cudaStream_t stream) {

  // TODO_HDR this is a sync point. We should try to make it non-blocking.
  // E.g. put it on the `stream`? Make it async? Avoid re-sending the matrix if
  // it hasn't changed from before (it likely didn't for the same video!!)?
  cudaMemcpyToSymbol(
      d_colorMatrix, colorMatrix, sizeof(float) * 12, 0, cudaMemcpyHostToDevice);

  int yPitchElements = yPitch / static_cast<int>(sizeof(uint16_t));
  int uvPitchElements = uvPitch / static_cast<int>(sizeof(uint16_t));
  int rgbPitchElements = rgbPitch / static_cast<int>(sizeof(uint16_t));
  int bitShift = 16 - bitDepth;

  // TODO_HDR: investigate perf implications of the block and grid size?
  dim3 block(16, 16);
  dim3 grid(
      (width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  p016ToRgb16Kernel<<<grid, block, 0, stream>>>(
      yPlane,
      uvPlane,
      rgbOutput,
      width,
      height,
      yPitchElements,
      uvPitchElements,
      rgbPitchElements,
      bitShift);
}

} // namespace facebook::torchcodec
