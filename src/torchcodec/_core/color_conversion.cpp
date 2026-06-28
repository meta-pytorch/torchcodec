// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "color_conversion.h"
#include "CUDACommon.h"
#include "StableABICompat.h"
#include "ValidationUtils.h"

namespace facebook::torchcodec {

/* clang-format off */
// Note: [YUV -> RGB Color Conversion]
//
// The color conversion matrix is derived from the color space's luma
// coefficients (kr, kg, kb) and the color range (full or limited).
//
// The forward (RGB -> YUV) transform is:
//   Y = kr*R + kg*G + kb*B
//   U = (B - Y) / (2*(1-kb))
//   V = (R - Y) / (2*(1-kr))
//
// Inverting gives the YUV -> RGB matrix. The 4th column of each row encodes
// all additive offsets (UV centering and, for limited range, the Y pedestal)
// so that the kernel can apply the matrix directly to raw sample values
// without any pre-processing.
//
// For full range:
//   Input Y in [0, maxVal], U/V in [0, maxVal] centered at maxVal/2+1.
//   Output RGB in [0, outScale].
//
// For limited (studio) range:
//   Y in [16*s, 235*s], U/V in [16*s, 240*s] where s = 2^(bitDepth-8).
//   Mapped to output RGB in [0, outScale].
//
// The same formula works for both NV12 (bitDepth=8, outScale=255) and P016
// (bitDepth=10 or 12, outScale=65535).
/* clang-format on */

LumaCoefficients get_luma_coefficients(AVColorSpace colorspace) {
  switch (colorspace) {
    case AVCOL_SPC_BT709:
      return {0.2126f, 0.7152f, 0.0722f};
    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:
      return {0.2627f, 0.6780f, 0.0593f};
    default:
      // BT.601: default for unspecified colorspace, matching FFmpeg's
      // swscale behavior (sws_getCoefficients(SWS_CS_DEFAULT) returns
      // BT.601 coefficients).
      return {0.299f, 0.587f, 0.114f};
  }
}

void compute_color_conversion_matrix(
    AVColorSpace colorspace,
    AVColorRange color_range,
    int bit_depth,
    float out_scale,
    float out_matrix[3][4]) {
  auto [kr, kg, kb] = get_luma_coefficients(colorspace);

  float v_scale = 2.0f * (1.0f - kr);
  float u_scale = 2.0f * (1.0f - kb);
  float gu_coeff = -(2.0f * kb * (1.0f - kb)) / kg;
  float gv_coeff = -(2.0f * kr * (1.0f - kr)) / kg;

  float max_val = static_cast<float>((1 << bit_depth) - 1);

  bool is_full_range = (color_range == AVCOL_RANGE_JPEG);

  if (is_full_range) {
    float y_scale = out_scale / max_val;
    float uv_center = static_cast<float>(1 << (bit_depth - 1));

    out_matrix[0][0] = y_scale;
    out_matrix[0][1] = 0.0f;
    out_matrix[0][2] = v_scale * out_scale / max_val;
    out_matrix[0][3] = -v_scale * uv_center * out_scale / max_val;

    out_matrix[1][0] = y_scale;
    out_matrix[1][1] = gu_coeff * out_scale / max_val;
    out_matrix[1][2] = gv_coeff * out_scale / max_val;
    out_matrix[1][3] = -(gu_coeff + gv_coeff) * uv_center * out_scale / max_val;

    out_matrix[2][0] = y_scale;
    out_matrix[2][1] = u_scale * out_scale / max_val;
    out_matrix[2][2] = 0.0f;
    out_matrix[2][3] = -u_scale * uv_center * out_scale / max_val;
  } else {
    float s = static_cast<float>(1 << (bit_depth - 8));
    float y_off = 16.0f * s;
    float y_range = 219.0f * s;
    float uv_off = 128.0f * s;
    float uv_range = 224.0f * s;

    float y_coeff = out_scale / y_range;
    float uv_coeff_u = out_scale / uv_range;
    float uv_coeff_v = out_scale / uv_range;

    out_matrix[0][0] = y_coeff;
    out_matrix[0][1] = 0.0f;
    out_matrix[0][2] = v_scale * uv_coeff_v;
    out_matrix[0][3] = -y_coeff * y_off - v_scale * uv_coeff_v * uv_off;

    out_matrix[1][0] = y_coeff;
    out_matrix[1][1] = gu_coeff * uv_coeff_u;
    out_matrix[1][2] = gv_coeff * uv_coeff_v;
    out_matrix[1][3] = -y_coeff * y_off - gu_coeff * uv_coeff_u * uv_off -
        gv_coeff * uv_coeff_v * uv_off;

    out_matrix[2][0] = y_coeff;
    out_matrix[2][1] = u_scale * uv_coeff_u;
    out_matrix[2][2] = 0.0f;
    out_matrix[2][3] = -y_coeff * y_off - u_scale * uv_coeff_u * uv_off;
  }
}

void maybe_update_color_matrix(
    CachedColorMatrix& cached_color_matrix,
    AVColorSpace colorspace,
    AVColorRange color_range,
    int bit_depth,
    float out_scale) {
  if (cached_color_matrix.valid &&
      cached_color_matrix.colorspace == colorspace &&
      cached_color_matrix.color_range == color_range &&
      cached_color_matrix.bit_depth == bit_depth &&
      cached_color_matrix.out_scale == out_scale) {
    return;
  }

  compute_color_conversion_matrix(
      colorspace,
      color_range,
      bit_depth,
      out_scale,
      cached_color_matrix.matrix);
  cached_color_matrix.colorspace = colorspace;
  cached_color_matrix.color_range = color_range;
  cached_color_matrix.bit_depth = bit_depth;
  cached_color_matrix.out_scale = out_scale;
  cached_color_matrix.valid = true;
}

void compute_rgb_to_yuv_matrix(
    AVColorSpace colorspace,
    AVColorRange color_range,
    float out_matrix[3][4]) {
  auto [kr, kg, kb] = get_luma_coefficients(colorspace);

  float u_scale = 2.0f * (1.0f - kb);
  float v_scale = 2.0f * (1.0f - kr);

  // Full-range RGB [0,255] -> YUV forward matrix
  // Y = kr*R + kg*G + kb*B
  // U = (-kr*R - kg*G + (1-kb)*B) / uScale
  // V = ((1-kr)*R - kg*G - kb*B) / vScale
  float y_row[3] = {kr, kg, kb};
  float u_row[3] = {-kr / u_scale, -kg / u_scale, (1.0f - kb) / u_scale};
  float v_row[3] = {(1.0f - kr) / v_scale, -kg / v_scale, -kb / v_scale};

  bool is_full_range = (color_range == AVCOL_RANGE_JPEG);

  if (is_full_range) {
    for (int i = 0; i < 3; i++) {
      out_matrix[0][i] = y_row[i];
      out_matrix[1][i] = u_row[i];
      out_matrix[2][i] = v_row[i];
    }
    out_matrix[0][3] = 0.0f;
    out_matrix[1][3] = 128.0f;
    out_matrix[2][3] = 128.0f;
  } else {
    // Limited range: Y scaled to [16, 235], UV scaled to [16, 240]
    float y_scale = 219.0f / 255.0f;
    float uv_limited_scale = 224.0f / 255.0f;
    for (int i = 0; i < 3; i++) {
      out_matrix[0][i] = y_row[i] * y_scale;
      out_matrix[1][i] = u_row[i] * uv_limited_scale;
      out_matrix[2][i] = v_row[i] * uv_limited_scale;
    }
    out_matrix[0][3] = 16.0f;
    out_matrix[1][3] = 128.0f;
    out_matrix[2][3] = 128.0f;
  }
}

torch::stable::Tensor convert_yuv_frame_to_rgb(
    UniqueAVFrame& av_frame,
    const StableDevice& device,
    cudaStream_t nvdec_stream,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor,
    const FrameDims& output_dims,
    bool is_p016,
    int bit_depth,
    CachedColorMatrix& cached_color_matrix) {
  float out_scale = is_p016 ? 65535.0f : 255.0f;
  OutputDtype out_dtype = is_p016 ? OutputDtype::FLOAT32 : OutputDtype::UINT8;

  // Dimensions may be odd (NVDEC display area for VP9 etc.). NV12/P016
  // color conversion requires even dimensions, so we round up to even
  // for the kernel, then crop to outputDims.
  int even_height = round_up_to_even(av_frame->height);
  int even_width = round_up_to_even(av_frame->width);

  int out_height = output_dims.height;
  int out_width = output_dims.width;
  bool needs_crop = (out_height != even_height) || (out_width != even_width);

  torch::stable::Tensor dst;
  if (needs_crop) {
    dst = allocate_empty_hwc_tensor(
        FrameDims(even_height, even_width), device, out_dtype);
  } else if (pre_allocated_output_tensor.has_value()) {
    dst = pre_allocated_output_tensor.value();
  } else {
    dst = allocate_empty_hwc_tensor(
        FrameDims(out_height, out_width), device, out_dtype);
  }

  cudaStream_t stream = get_current_cuda_stream(device.index());
  sync_streams(
      /*runningStream=*/nvdec_stream, /*waitingStream=*/stream);

  maybe_update_color_matrix(
      cached_color_matrix,
      av_frame->colorspace,
      av_frame->color_range,
      bit_depth,
      out_scale);

  if (is_p016) {
    launch_p016_to_rgb16_kernel(
        reinterpret_cast<const uint16_t*>(av_frame->data[0]),
        reinterpret_cast<const uint16_t*>(av_frame->data[1]),
        dst.mutable_data_ptr<uint16_t>(),
        even_width,
        even_height,
        av_frame->linesize[0],
        av_frame->linesize[1],
        validate_int64_to_int(dst.stride(0) * 2, "dst.stride(0)*2"),
        bit_depth,
        cached_color_matrix.matrix,
        stream);
  } else {
    launch_nv12_to_rgb_kernel(
        av_frame->data[0],
        av_frame->data[1],
        dst.mutable_data_ptr<uint8_t>(),
        even_width,
        even_height,
        av_frame->linesize[0],
        av_frame->linesize[1],
        validate_int64_to_int(dst.stride(0), "dst.stride(0)"),
        cached_color_matrix.matrix,
        stream);
  }

  if (needs_crop) {
    if (out_height != even_height) {
      dst = torch::stable::narrow(dst, /*dim=*/0, /*start=*/0, out_height);
    }
    if (out_width != even_width) {
      dst = torch::stable::narrow(dst, /*dim=*/1, /*start=*/0, out_width);
      dst = torch::stable::contiguous(dst);
    }
    if (pre_allocated_output_tensor.has_value()) {
      torch::stable::copy_(pre_allocated_output_tensor.value(), dst);
      return pre_allocated_output_tensor.value();
    }
    return dst;
  }
  return dst;
}

} // namespace facebook::torchcodec
