// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "color_conversion.h"

namespace facebook::torchcodec {

// NV12 and P016 are semi-planar YUV 4:2:0 formats. Chroma subsampling is
// 4:2:0 so U,V are subsampled by 2 in both H and W dimensions. This means a
// single UV pair is responsible for a 2x2 pixel block.
// The Y plane is YYYYY... where each Y corresponds to one pixel.
// The UV plane is interleaved as UVUVUVUV...
// NV12 uses 8-bit samples (uint8_t), i.e. the Y, U and V values are 8 bits
// each. P016 uses 16-bit samples (uint16_t) with actual data in the most
// significant bits (right-shifted by 16 - bitDepth).

// Wraps the 3x4 color-conversion matrix so it can be passed as a kernel param,
// to ensure thread-safety (previous implem had it __global__, which wasn't
// safe)
struct ColorMatrix {
  float m[3][4];
};

// Takes a pair of consecutive Y values, a pair of UV values, and writes the
// corresponding two RGB values. Instead of writing each ra ga ba and rb gb bb
// separately, they are written in pairs as Vec2T for faster writes.
//
// T is the sample type: uint8_t for NV12, uint16_t for P016.
// Vec2T is the corresponding 2-element vector: uchar2 (2 uint8 values for NV12)
// or ushort2 (2 uint16 values for P016).
template <typename T, typename Vec2T>
__device__ void write_pair_of_rgb_pixels(
    Vec2T yayb,
    float u,
    float v,
    T* rgb_plane_to_write,
    int bit_shift,
    const ColorMatrix& cm) {
  constexpr float clamp_max = sizeof(T) == 1 ? 255.0f : 65535.0f;

  float ya = static_cast<float>(yayb.x >> bit_shift);
  float yb = static_cast<float>(yayb.y >> bit_shift);

  float ra = cm.m[0][0] * ya + cm.m[0][1] * u + cm.m[0][2] * v + cm.m[0][3];
  float ga = cm.m[1][0] * ya + cm.m[1][1] * u + cm.m[1][2] * v + cm.m[1][3];

  Vec2T raga = {
      static_cast<T>(fminf(fmaxf(ra, 0.0f), clamp_max)),
      static_cast<T>(fminf(fmaxf(ga, 0.0f), clamp_max))};
  *(reinterpret_cast<Vec2T*>(&rgb_plane_to_write[0])) = raga;

  float ba = cm.m[2][0] * ya + cm.m[2][1] * u + cm.m[2][2] * v + cm.m[2][3];
  float rb = cm.m[0][0] * yb + cm.m[0][1] * u + cm.m[0][2] * v + cm.m[0][3];
  Vec2T barb = {
      static_cast<T>(fminf(fmaxf(ba, 0.0f), clamp_max)),
      static_cast<T>(fminf(fmaxf(rb, 0.0f), clamp_max))};
  *(reinterpret_cast<Vec2T*>(&rgb_plane_to_write[2])) = barb;

  float gb = cm.m[1][0] * yb + cm.m[1][1] * u + cm.m[1][2] * v + cm.m[1][3];
  float bb = cm.m[2][0] * yb + cm.m[2][1] * u + cm.m[2][2] * v + cm.m[2][3];
  Vec2T gbbb = {
      static_cast<T>(fminf(fmaxf(gb, 0.0f), clamp_max)),
      static_cast<T>(fminf(fmaxf(bb, 0.0f), clamp_max))};
  *(reinterpret_cast<Vec2T*>(&rgb_plane_to_write[4])) = gbbb;
}

// Takes the Y and UV plane as input, applies the color-conversion matrix and
// fills the RGB plane as output.
// Each thread, i.e. each invocation of yuvToRgbKernel, processes a 2x2 block
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
// We use Vec2T vectorized loads to read {y1,y2} and {y3,y4} in two
// reads, and {U,V} in one read. Then we apply the color matrix to produce
// 4 RGB pixels.
template <typename T, typename Vec2T>
__global__ void yuv_to_rgb_kernel(
    // __restrict__ tells the compiler those pointers never overlap with each
    // other so it can optimize read and writes more aggressively.
    const T* __restrict__ y_plane,
    const T* __restrict__ uv_plane,
    T* __restrict__ rgb_output,
    int width,
    int height,
    int y_pitch_elements,
    int uv_pitch_elements,
    int rgb_pitch_elements,
    int bit_shift,
    const ColorMatrix cm) {
  // The kernel operates on 2x2 blocks, so it's called H / 2 * W / 2 times.
  // We have to multiply back by 2 to retrieve the output pixel coordinates x
  // and y.
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  if (x + 1 >= width || y + 1 >= height) {
    return;
  }

  // Vec2T stores two values in .x and .y
  // Here, we read the UV pair in one instruction.
  int uv_idx = (y / 2) * uv_pitch_elements + x;
  Vec2T uv = *reinterpret_cast<const Vec2T*>(&uv_plane[uv_idx]);
  float u = static_cast<float>(uv.x >> bit_shift);
  float v = static_cast<float>(uv.y >> bit_shift);

  // Similarly, we can read 4 Y values in 2 reads
  Vec2T y1y2 =
      *reinterpret_cast<const Vec2T*>(&y_plane[y * y_pitch_elements + x]);
  Vec2T y3y4 =
      *reinterpret_cast<const Vec2T*>(&y_plane[(y + 1) * y_pitch_elements + x]);

  T* rgb_plane_to_write = rgb_output + y * rgb_pitch_elements + x * 3;
  write_pair_of_rgb_pixels<T, Vec2T>(
      y1y2, u, v, rgb_plane_to_write, bit_shift, cm);
  rgb_plane_to_write += rgb_pitch_elements; // go to next line
  write_pair_of_rgb_pixels<T, Vec2T>(
      y3y4, u, v, rgb_plane_to_write, bit_shift, cm);
}

void launch_nv12_to_rgb_kernel(
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    uint8_t* rgb_output,
    int width,
    int height,
    int y_pitch,
    int uv_pitch,
    int rgb_pitch,
    const float color_matrix[3][4],
    cudaStream_t stream) {
  const auto& cm = *reinterpret_cast<const ColorMatrix*>(color_matrix);

  dim3 block(32, 2);
  dim3 grid(
      (width / 2 + block.x - 1) / block.x,
      (height / 2 + block.y - 1) / block.y);

  yuv_to_rgb_kernel<uint8_t, uchar2><<<grid, block, 0, stream>>>(
      y_plane,
      uv_plane,
      rgb_output,
      width,
      height,
      y_pitch,
      uv_pitch,
      rgb_pitch,
      0, // bitShift = 0 for NV12
      cm);
}

void launch_p016_to_rgb16_kernel(
    const uint16_t* y_plane,
    const uint16_t* uv_plane,
    uint16_t* rgb_output,
    int width,
    int height,
    int y_pitch,
    int uv_pitch,
    int rgb_pitch,
    int bit_depth,
    const float color_matrix[3][4],
    cudaStream_t stream) {
  const auto& cm = *reinterpret_cast<const ColorMatrix*>(color_matrix);

  int y_pitch_elements = y_pitch / static_cast<int>(sizeof(uint16_t));
  int uv_pitch_elements = uv_pitch / static_cast<int>(sizeof(uint16_t));
  int rgb_pitch_elements = rgb_pitch / static_cast<int>(sizeof(uint16_t));
  int bit_shift = 16 - bit_depth;

  dim3 block(32, 2);
  dim3 grid(
      (width / 2 + block.x - 1) / block.x,
      (height / 2 + block.y - 1) / block.y);

  yuv_to_rgb_kernel<uint16_t, ushort2><<<grid, block, 0, stream>>>(
      y_plane,
      uv_plane,
      rgb_output,
      width,
      height,
      y_pitch_elements,
      uv_pitch_elements,
      rgb_pitch_elements,
      bit_shift,
      cm);
}

// RGB -> NV12 kernel for encoding.
// Each thread processes a 2x2 block: computes Y for all 4 pixels,
// averages U and V across the block, and writes to the NV12 planes.
__global__ void rgb_to_nv12_kernel(
    const uint8_t* __restrict__ rgb_input,
    uint8_t* __restrict__ y_plane,
    uint8_t* __restrict__ uv_plane,
    int width,
    int height,
    int rgb_pitch_elements,
    int y_pitch_elements,
    int uv_pitch_elements,
    const ColorMatrix cm) {
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  if (x + 1 >= width || y + 1 >= height) {
    return;
  }

  // Read 4 RGB pixels from 2x2 block
  const uint8_t* row0 = rgb_input + y * rgb_pitch_elements + x * 3;
  const uint8_t* row1 = rgb_input + (y + 1) * rgb_pitch_elements + x * 3;

  float r00 = row0[0], g00 = row0[1], b00 = row0[2];
  float r01 = row0[3], g01 = row0[4], b01 = row0[5];
  float r10 = row1[0], g10 = row1[1], b10 = row1[2];
  float r11 = row1[3], g11 = row1[4], b11 = row1[5];

  // Compute Y for all 4 pixels
  float y00 =
      cm.m[0][0] * r00 + cm.m[0][1] * g00 + cm.m[0][2] * b00 + cm.m[0][3];
  float y01 =
      cm.m[0][0] * r01 + cm.m[0][1] * g01 + cm.m[0][2] * b01 + cm.m[0][3];
  float y10 =
      cm.m[0][0] * r10 + cm.m[0][1] * g10 + cm.m[0][2] * b10 + cm.m[0][3];
  float y11 =
      cm.m[0][0] * r11 + cm.m[0][1] * g11 + cm.m[0][2] * b11 + cm.m[0][3];

  // Write Y plane
  uchar2 y_row0 = {
      static_cast<uint8_t>(fminf(fmaxf(roundf(y00), 0.0f), 255.0f)),
      static_cast<uint8_t>(fminf(fmaxf(roundf(y01), 0.0f), 255.0f))};
  *(reinterpret_cast<uchar2*>(&y_plane[y * y_pitch_elements + x])) = y_row0;

  uchar2 y_row1 = {
      static_cast<uint8_t>(fminf(fmaxf(roundf(y10), 0.0f), 255.0f)),
      static_cast<uint8_t>(fminf(fmaxf(roundf(y11), 0.0f), 255.0f))};
  *(reinterpret_cast<uchar2*>(&y_plane[(y + 1) * y_pitch_elements + x])) =
      y_row1;

  // Compute U,V for all 4 pixels and average for 4:2:0 subsampling
  float u00 =
      cm.m[1][0] * r00 + cm.m[1][1] * g00 + cm.m[1][2] * b00 + cm.m[1][3];
  float u01 =
      cm.m[1][0] * r01 + cm.m[1][1] * g01 + cm.m[1][2] * b01 + cm.m[1][3];
  float u10 =
      cm.m[1][0] * r10 + cm.m[1][1] * g10 + cm.m[1][2] * b10 + cm.m[1][3];
  float u11 =
      cm.m[1][0] * r11 + cm.m[1][1] * g11 + cm.m[1][2] * b11 + cm.m[1][3];

  float v00 =
      cm.m[2][0] * r00 + cm.m[2][1] * g00 + cm.m[2][2] * b00 + cm.m[2][3];
  float v01 =
      cm.m[2][0] * r01 + cm.m[2][1] * g01 + cm.m[2][2] * b01 + cm.m[2][3];
  float v10 =
      cm.m[2][0] * r10 + cm.m[2][1] * g10 + cm.m[2][2] * b10 + cm.m[2][3];
  float v11 =
      cm.m[2][0] * r11 + cm.m[2][1] * g11 + cm.m[2][2] * b11 + cm.m[2][3];

  float u_avg = (u00 + u01 + u10 + u11) * 0.25f;
  float v_avg = (v00 + v01 + v10 + v11) * 0.25f;

  int uv_idx = (y / 2) * uv_pitch_elements + x;
  uchar2 uv_pair = {
      static_cast<uint8_t>(fminf(fmaxf(roundf(u_avg), 0.0f), 255.0f)),
      static_cast<uint8_t>(fminf(fmaxf(roundf(v_avg), 0.0f), 255.0f))};
  *(reinterpret_cast<uchar2*>(&uv_plane[uv_idx])) = uv_pair;
}

void launch_rgb_to_nv12_kernel(
    const uint8_t* rgb_input,
    uint8_t* y_plane,
    uint8_t* uv_plane,
    int width,
    int height,
    int rgb_pitch,
    int y_pitch,
    int uv_pitch,
    const float color_matrix[3][4],
    cudaStream_t stream) {
  const auto& cm = *reinterpret_cast<const ColorMatrix*>(color_matrix);

  dim3 block(32, 2);
  dim3 grid(
      (width / 2 + block.x - 1) / block.x,
      (height / 2 + block.y - 1) / block.y);

  rgb_to_nv12_kernel<<<grid, block, 0, stream>>>(
      rgb_input,
      y_plane,
      uv_plane,
      width,
      height,
      rgb_pitch,
      y_pitch,
      uv_pitch,
      cm);
}

} // namespace facebook::torchcodec
