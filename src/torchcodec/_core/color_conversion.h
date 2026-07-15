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

LumaCoefficients get_luma_coefficients(AVColorSpace colorspace);

struct CachedColorMatrix {
  AVColorSpace colorspace = AVCOL_SPC_UNSPECIFIED;
  AVColorRange color_range = AVCOL_RANGE_UNSPECIFIED;
  int bit_depth = 0;
  float out_scale = 0;
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
void compute_color_conversion_matrix(
    AVColorSpace colorspace,
    AVColorRange color_range,
    int bit_depth,
    float out_scale,
    float out_matrix[3][4]);

void maybe_update_color_matrix(
    CachedColorMatrix& cached_color_matrix,
    AVColorSpace colorspace,
    AVColorRange color_range,
    int bit_depth,
    float out_scale);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

// Convert a YUV frame (NV12 or P016) on GPU to an interleaved RGB tensor.
//
// isP016: true for P016 (uint16 I/O), false for NV12 (uint8 I/O).
// bitDepth: 8 for NV12, actual bit depth (10 or 12) for P016.
// outputDims: desired output size; if the frame was rounded up to even
//   dimensions, the result is cropped back to outputDims.
torch::stable::Tensor convert_yuv_frame_to_rgb(
    UniqueAVFrame& av_frame,
    const StableDevice& device,
    cudaStream_t nvdec_stream,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor,
    const FrameDims& output_dims,
    bool is_p016,
    int bit_depth,
    CachedColorMatrix& cached_color_matrix);

// Compute the RGB -> YUV color conversion matrix (for encoding).
// The matrix operates on uint8 RGB values [0, 255] and produces Y, U, V
// values. The 4th column encodes offsets (Y pedestal for limited range,
// UV centering at 128).
void compute_rgb_to_yuv_matrix(
    AVColorSpace colorspace,
    AVColorRange color_range,
    float out_matrix[3][4]);

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
    cudaStream_t stream);

} // namespace facebook::torchcodec
