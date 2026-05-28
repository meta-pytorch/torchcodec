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

// For a *single* pixel, applies the color-conversion matrix to the YUV values
// and writes the result as uint16 in rgbOutput
__device__ void computeRGBPixel(float y, float u, float v, uint16_t* rgbOutput){
  float r = d_colorMatrix[0][0] * y + d_colorMatrix[0][1] * u +
      d_colorMatrix[0][2] * v + d_colorMatrix[0][3];
  float g = d_colorMatrix[1][0] * y + d_colorMatrix[1][1] * u +
      d_colorMatrix[1][2] * v + d_colorMatrix[1][3];
  float b = d_colorMatrix[2][0] * y + d_colorMatrix[2][1] * u +
      d_colorMatrix[2][2] * v + d_colorMatrix[2][3];

  rgbOutput[0] =
      static_cast<uint16_t>(fminf(fmaxf(r, 0.0f), 65535.0f));
  rgbOutput[1] =
      static_cast<uint16_t>(fminf(fmaxf(g, 0.0f), 65535.0f));
  rgbOutput[2] =
      static_cast<uint16_t>(fminf(fmaxf(b, 0.0f), 65535.0f));

}

// Takes the Y and UV plane as input, applies the color-conversion matrix and
// fills the RGB plane as output.
// Each thread, i.e. each invocation of p016ToRgb16Kernel, processes a 2x2 block
// of pixels: this optimizes the UV plan reads, since each UV pair is
// responsible for a 2x2 block.
//
//   Y plane (one value per pixel):       UV plane (one pair per 2x2 block):
//   +------+------+                      +----------+
//   |  y1  |  y2  |  row y               |  u  | v  |
//   +------+------+                      +----------+
//   |  y3  |  y4  |  row y+1
//   +------+------+
//   col x    col x+1
//
// We use ushort2 vectorized loads to read {y1,y2} and {y3,y4} in two
// 32-bit reads, and {U,V} in one 32-bit read. Then we apply the color
// matrix to produce 4 RGB pixels.
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
  // The kernel operates on 2x2 blocks, so it's called H / 2 * W / 2 times.
  // We have to multiply back by 2 to retrieve the output pixel coordinates x
  // and y.
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  if (x + 1 >= width || y + 1 >= height) {
    return;
  }

  // ushort2 stores two uint16 values in .x and .y
  // Here, we read the UV pair in one instruction. Both U and V are 16bits.
  int uvIdx = (y / 2) * uvPitchElements + x;
  ushort2 uv= *reinterpret_cast<const ushort2*>(&uvPlane[uvIdx]);
  float u = static_cast<float>(uv.x >> bitShift);
  float v = static_cast<float>(uv.y >> bitShift);

  // Similarly, we can read 4 Y values in 2 reads
  ushort2 y1y2 = *reinterpret_cast<const ushort2*>(&yPlane[y * yPitchElements + x]);
  ushort2 y3y4 = *reinterpret_cast<const ushort2*>(&yPlane[(y + 1) * yPitchElements + x]);

  float y1 = static_cast<float>(y1y2.x >> bitShift);
  float y2 = static_cast<float>(y1y2.y >> bitShift);
  int rgbIdx = y * rgbPitchElements + x * 3;
  computeRGBPixel(y1, u, v, rgbOutput + rgbIdx);
  computeRGBPixel(y2, u, v, rgbOutput + rgbIdx + 3);

  float y3 = static_cast<float>(y3y4.x >> bitShift);
  float y4 = static_cast<float>(y3y4.y >> bitShift);
  rgbIdx = (y + 1) * rgbPitchElements + x * 3;
  computeRGBPixel(y3, u, v, rgbOutput + rgbIdx );
  computeRGBPixel(y4, u, v, rgbOutput + rgbIdx + 3);
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
    bool colorMatrixChanged,
    cudaStream_t stream) {

  if (colorMatrixChanged) {
    // We only send the color-matrix from CPU to GPU if it changed.
    // In practice it probably doesn't impact perf that much since decoding is
    // bottlenecked by NVDEC's frame mapping, not color-conversion. But it's a
    // simple optimization, so we do it anyway.
    cudaMemcpyToSymbol(
        d_colorMatrix, colorMatrix, sizeof(float) * 12, 0,
        cudaMemcpyHostToDevice);
  }

  int yPitchElements = yPitch / static_cast<int>(sizeof(uint16_t));
  int uvPitchElements = uvPitch / static_cast<int>(sizeof(uint16_t));
  int rgbPitchElements = rgbPitch / static_cast<int>(sizeof(uint16_t));
  int bitShift = 16 - bitDepth;

  // TODO_HDR: investigate perf implications of the block and grid size?
  dim3 block(16, 16);
  dim3 grid(
      (width / 2 + block.x - 1) / block.x, (height / 2 + block.y - 1) / block.y);

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
