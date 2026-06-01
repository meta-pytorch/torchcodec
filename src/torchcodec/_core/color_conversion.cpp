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

LumaCoefficients getLumaCoefficients(AVColorSpace colorspace) {
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

void computeColorConversionMatrix(
    AVColorSpace colorspace,
    AVColorRange colorRange,
    int bitDepth,
    float outScale,
    float outMatrix[3][4]) {
  auto [kr, kg, kb] = getLumaCoefficients(colorspace);

  float vScale = 2.0f * (1.0f - kr);
  float uScale = 2.0f * (1.0f - kb);
  float guCoeff = -(2.0f * kb * (1.0f - kb)) / kg;
  float gvCoeff = -(2.0f * kr * (1.0f - kr)) / kg;

  float maxVal = static_cast<float>((1 << bitDepth) - 1);

  bool isFullRange = (colorRange == AVCOL_RANGE_JPEG);

  if (isFullRange) {
    float yScale = outScale / maxVal;
    float uvCenter = static_cast<float>(1 << (bitDepth - 1));

    outMatrix[0][0] = yScale;
    outMatrix[0][1] = 0.0f;
    outMatrix[0][2] = vScale * outScale / maxVal;
    outMatrix[0][3] = -vScale * uvCenter * outScale / maxVal;

    outMatrix[1][0] = yScale;
    outMatrix[1][1] = guCoeff * outScale / maxVal;
    outMatrix[1][2] = gvCoeff * outScale / maxVal;
    outMatrix[1][3] = -(guCoeff + gvCoeff) * uvCenter * outScale / maxVal;

    outMatrix[2][0] = yScale;
    outMatrix[2][1] = uScale * outScale / maxVal;
    outMatrix[2][2] = 0.0f;
    outMatrix[2][3] = -uScale * uvCenter * outScale / maxVal;
  } else {
    float s = static_cast<float>(1 << (bitDepth - 8));
    float yOff = 16.0f * s;
    float yRange = 219.0f * s;
    float uvOff = 128.0f * s;
    float uvRange = 224.0f * s;

    float yCoeff = outScale / yRange;
    float uvCoeff_u = outScale / uvRange;
    float uvCoeff_v = outScale / uvRange;

    outMatrix[0][0] = yCoeff;
    outMatrix[0][1] = 0.0f;
    outMatrix[0][2] = vScale * uvCoeff_v;
    outMatrix[0][3] = -yCoeff * yOff - vScale * uvCoeff_v * uvOff;

    outMatrix[1][0] = yCoeff;
    outMatrix[1][1] = guCoeff * uvCoeff_u;
    outMatrix[1][2] = gvCoeff * uvCoeff_v;
    outMatrix[1][3] = -yCoeff * yOff - guCoeff * uvCoeff_u * uvOff -
        gvCoeff * uvCoeff_v * uvOff;

    outMatrix[2][0] = yCoeff;
    outMatrix[2][1] = uScale * uvCoeff_u;
    outMatrix[2][2] = 0.0f;
    outMatrix[2][3] = -yCoeff * yOff - uScale * uvCoeff_u * uvOff;
  }
}

void maybeUpdateColorMatrix(
    CachedColorMatrix& cachedColorMatrix,
    AVColorSpace colorspace,
    AVColorRange colorRange,
    int bitDepth,
    float outScale) {
  if (cachedColorMatrix.valid && cachedColorMatrix.colorspace == colorspace &&
      cachedColorMatrix.colorRange == colorRange &&
      cachedColorMatrix.bitDepth == bitDepth &&
      cachedColorMatrix.outScale == outScale) {
    return;
  }

  computeColorConversionMatrix(
      colorspace, colorRange, bitDepth, outScale, cachedColorMatrix.matrix);
  cachedColorMatrix.colorspace = colorspace;
  cachedColorMatrix.colorRange = colorRange;
  cachedColorMatrix.bitDepth = bitDepth;
  cachedColorMatrix.outScale = outScale;
  cachedColorMatrix.valid = true;
}

void computeRGBToYUVMatrix(
    AVColorSpace colorspace,
    AVColorRange colorRange,
    float outMatrix[3][4]) {
  auto [kr, kg, kb] = getLumaCoefficients(colorspace);

  float uScale = 2.0f * (1.0f - kb);
  float vScale = 2.0f * (1.0f - kr);

  // Full-range RGB [0,255] -> YUV forward matrix
  // Y = kr*R + kg*G + kb*B
  // U = (-kr*R - kg*G + (1-kb)*B) / uScale
  // V = ((1-kr)*R - kg*G - kb*B) / vScale
  float yRow[3] = {kr, kg, kb};
  float uRow[3] = {-kr / uScale, -kg / uScale, (1.0f - kb) / uScale};
  float vRow[3] = {(1.0f - kr) / vScale, -kg / vScale, -kb / vScale};

  bool isFullRange = (colorRange == AVCOL_RANGE_JPEG);

  if (isFullRange) {
    for (int i = 0; i < 3; i++) {
      outMatrix[0][i] = yRow[i];
      outMatrix[1][i] = uRow[i];
      outMatrix[2][i] = vRow[i];
    }
    outMatrix[0][3] = 0.0f;
    outMatrix[1][3] = 128.0f;
    outMatrix[2][3] = 128.0f;
  } else {
    // Limited range: Y scaled to [16, 235], UV scaled to [16, 240]
    float yScale = 219.0f / 255.0f;
    float uvLimitedScale = 224.0f / 255.0f;
    for (int i = 0; i < 3; i++) {
      outMatrix[0][i] = yRow[i] * yScale;
      outMatrix[1][i] = uRow[i] * uvLimitedScale;
      outMatrix[2][i] = vRow[i] * uvLimitedScale;
    }
    outMatrix[0][3] = 16.0f;
    outMatrix[1][3] = 128.0f;
    outMatrix[2][3] = 128.0f;
  }
}

torch::stable::Tensor convertYUVFrameToRGB(
    UniqueAVFrame& avFrame,
    const StableDevice& device,
    cudaStream_t nvdecStream,
    std::optional<torch::stable::Tensor> preAllocatedOutputTensor,
    const FrameDims& outputDims,
    bool isP016,
    int bitDepth,
    CachedColorMatrix& cachedColorMatrix) {
  float outScale = isP016 ? 65535.0f : 255.0f;
  OutputDtype outDtype = isP016 ? OutputDtype::FLOAT32 : OutputDtype::UINT8;

  // Dimensions may be odd (NVDEC display area for VP9 etc.). NV12/P016
  // color conversion requires even dimensions, so we round up to even
  // for the kernel, then crop to outputDims.
  int evenHeight = roundUpToEven(avFrame->height);
  int evenWidth = roundUpToEven(avFrame->width);

  int outHeight = outputDims.height;
  int outWidth = outputDims.width;
  bool needsCrop = (outHeight != evenHeight) || (outWidth != evenWidth);

  torch::stable::Tensor dst;
  if (needsCrop) {
    dst = allocateEmptyHWCTensor(
        FrameDims(evenHeight, evenWidth), device, outDtype);
  } else if (preAllocatedOutputTensor.has_value()) {
    dst = preAllocatedOutputTensor.value();
  } else {
    dst = allocateEmptyHWCTensor(
        FrameDims(outHeight, outWidth), device, outDtype);
  }

  cudaStream_t stream = getCurrentCudaStream(device.index());
  syncStreams(
      /*runningStream=*/nvdecStream, /*waitingStream=*/stream);

  maybeUpdateColorMatrix(
      cachedColorMatrix,
      avFrame->colorspace,
      avFrame->color_range,
      bitDepth,
      outScale);

  if (isP016) {
    launchP016ToRGB16Kernel(
        reinterpret_cast<const uint16_t*>(avFrame->data[0]),
        reinterpret_cast<const uint16_t*>(avFrame->data[1]),
        dst.mutable_data_ptr<uint16_t>(),
        evenWidth,
        evenHeight,
        avFrame->linesize[0],
        avFrame->linesize[1],
        validateInt64ToInt(dst.stride(0) * 2, "dst.stride(0)*2"),
        bitDepth,
        cachedColorMatrix.matrix,
        stream);
  } else {
    launchNV12ToRGBKernel(
        avFrame->data[0],
        avFrame->data[1],
        dst.mutable_data_ptr<uint8_t>(),
        evenWidth,
        evenHeight,
        avFrame->linesize[0],
        avFrame->linesize[1],
        validateInt64ToInt(dst.stride(0), "dst.stride(0)"),
        cachedColorMatrix.matrix,
        stream);
  }

  if (needsCrop) {
    if (outHeight != evenHeight) {
      dst = torch::stable::narrow(dst, /*dim=*/0, /*start=*/0, outHeight);
    }
    if (outWidth != evenWidth) {
      dst = torch::stable::narrow(dst, /*dim=*/1, /*start=*/0, outWidth);
      dst = torch::stable::contiguous(dst);
    }
    if (preAllocatedOutputTensor.has_value()) {
      torch::stable::copy_(preAllocatedOutputTensor.value(), dst);
      return preAllocatedOutputTensor.value();
    }
    return dst;
  }
  return dst;
}

} // namespace facebook::torchcodec
