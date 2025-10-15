// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/CUDACommon.h"
#include "src/torchcodec/_core/Cache.h" // for PerGpuCache

namespace facebook::torchcodec {

namespace {

// Set to -1 to have an infinitely sized cache. Set it to 0 to disable caching.
// Set to a positive number to have a cache of that size.
const int MAX_CONTEXTS_PER_GPU_IN_CACHE = -1;

PerGpuCache<NppStreamContext> g_cached_npp_ctxs(
    MAX_CUDA_GPUS,
    MAX_CONTEXTS_PER_GPU_IN_CACHE);

} // namespace

void initializeCudaContextWithPytorch(const torch::Device& device) {
  // It is important for pytorch itself to create the cuda context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the cuda context.
  torch::Tensor dummyTensorForCudaInitialization = torch::zeros(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
}

/* clang-format off */
// Note: [YUV -> RGB Color Conversion, color space and color range]
//
// The frames we get from the decoder (FFmpeg decoder, or NVCUVID) are in YUV
// format. We need to convert them to RGB. This note attempts to describe this
// process. There may be some inaccuracies and approximations that experts will
// notice, but our goal is only to provide a good enough understanding of the
// process for torchcodec developers to implement and maintain it.
// On CPU, filtergraph and swscale handle everything for us. With CUDA, we have
// to do a lot of the heavy lifting ourselves.
//
// Color space and color range
// ---------------------------
// Two main characteristics of a frame will affect the conversion process:
// 1. Color space: This basically defines what YUV values correspond to which
//    physical wavelength. No need to go into details here,the point is that
//    videos can come in different color spaces, the most common ones being
//    BT.601 and BT.709, but there are others.
//    In FFmpeg this is represented with AVColorSpace:
//    https://ffmpeg.org/doxygen/4.0/pixfmt_8h.html#aff71a069509a1ad3ff54d53a1c894c85
// 2. Color range: This defines the range of YUV values. There is:
//    - full range, also called PC range: AVCOL_RANGE_JPEG
//    - and the "limited" range, also called studio or TV range: AVCOL_RANGE_MPEG
//    https://ffmpeg.org/doxygen/4.0/pixfmt_8h.html#a3da0bf691418bc22c4bcbe6583ad589a
//
// Color space and color range are independent concepts, so we can have a BT.709
// with full range, and another one with limited range. Same for BT.601.
//
// In the first version of this note we'll focus on the full color range. It
// will later be updated to account for the limited range.
//
// Color conversion matrix
// -----------------------
// YUV -> RGB conversion is defined as the reverse process of the RGB -> YUV,
// So this is where we'll start.
// At the core of a RGB -> YUV conversion are the "luma coefficients", which are
// specific to a given color space and defined by the color space standard. In
// FFmpeg they can be found here:
// https://github.com/FFmpeg/FFmpeg/blob/7d606ef0ccf2946a4a21ab1ec23486cadc21864b/libavutil/csp.c#L46-L56
//
// For example, the BT.709 coefficients are: kr=0.2126, kg=0.7152, kb=0.0722
// Coefficients must sum to 1.
//
// Conventionally Y is in [0, 1] range, and U and V are in [-0.5, 0.5] range
// (that's mathematically, in practice they are represented in integer range).
// The conversion is defined as:
// https://en.wikipedia.org/wiki/YCbCr#R'G'B'_to_Y%E2%80%B2PbPr
// Y = kr*R + kg*G + kb*B
// U = (B - Y) * 0.5 / (1 - kb) = (B - Y) / u_scale where u_scale = 2 * (1 - kb)
// V = (R - Y) * 0.5 / (1 - kr) = (R - Y) / v_scale where v_scale = 2 * (1 - kr)
//
// Putting all this into matrix form, we get:
// [Y]   = [kr               kg            kb            ]  [R]
// [U]     [-kr/u_scale      -kg/u_scale   (1-kb)/u_scale]  [G]
// [V]     [(1-kr)/v_scale   -kg/v_scale   -kb)/v_scale  ]  [B]
//
//
// Now, to convert YUV to RGB, we just need to invert this matrix:
// ```py
// import torch
// kr, kg, kb = 0.2126, 0.7152, 0.0722  # BT.709  luma coefficients
// u_scale = 2 * (1 - kb)
// v_scale = 2 * (1 - kr)
//
// rgb_to_yuv = torch.tensor([
//     [kr, kg, kb],
//     [-kr/u_scale, -kg/u_scale, (1-kb)/u_scale],
//     [(1-kr)/v_scale, -kg/v_scale, -kb/v_scale]
// ])
//
// yuv_to_rgb_full = torch.linalg.inv(rgb_to_yuv)
// print("YUV->RGB matrix (Full Range):")
// print(yuv_to_rgb_full)
// ```
// And we get:
// tensor([[ 1.0000e+00, -3.3142e-09,  1.5748e+00],
//         [ 1.0000e+00, -1.8732e-01, -4.6812e-01],
//         [ 1.0000e+00,  1.8556e+00,  4.6231e-09]])
//
// Which matches https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion
//
// Color conversion in NPP
// -----------------------
// https://docs.nvidia.com/cuda/npp/image_color_conversion.html.
//
// NPP provides different ways to convert YUV to RGB:
// - pre-defined color conversion functions like
//   nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx and nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx
//   which are for BT.709 limited and full range, respectively.
// - generic color conversion functions that accept a custom color conversion
//   matrix, called ColorTwist, like nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx
//
// We use the pre-defined functions or the color twist functions depending on
// which one we find to be closer to the CPU results.
//
// The color twist functionality is *partially* described in a section named
// "YUVToRGBColorTwist". Importantly:
//
// - The `nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx` function takes the YUV data
//   and the color-conversion matrix as input. The function itself and the
//   matrix assume different ranges for YUV values:
// - The **matrix coefficient** must assume that Y is in [0, 1] and U,V are in
//   [-0.5, 0.5]. That's how we defined our matrix above.
// - The function `nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx` however expects all
//   of the input Y, U, V to be in [0, 255]. That's how the data comes out of
//   the decoder.
// - But *internally*, `nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx` needs U and V to
//   be centered around 0, i.e. in [-128, 127]. So we need to apply a -128
//   offset to U and V. Y doesn't need to be offset. The offset can be applied
//   by adding a 4th column to the matrix.
//
//
// So our conversion matrix becomes the following, with new offset column:
// tensor([[ 1.0000e+00, -3.3142e-09,  1.5748e+00,     0]
//         [ 1.0000e+00, -1.8732e-01, -4.6812e-01,     -128]
//         [ 1.0000e+00,  1.8556e+00,  4.6231e-09 ,    -128]])
//
// And that's what we need to pass for BT701, full range.
/* clang-format on */

// BT.709 full range color conversion matrix for YUV to RGB conversion.
// See Note [YUV -> RGB Color Conversion, color space and color range]
const Npp32f bt709FullRangeColorTwist[3][4] = {
    {1.0f, 0.0f, 1.5748f, 0.0f},
    {1.0f, -0.187324273f, -0.468124273f, -128.0f},
    {1.0f, 1.8556f, 0.0f, -128.0f}};

torch::Tensor convertNV12FrameToRGB(
    UniqueAVFrame& avFrame,
    const torch::Device& device,
    const UniqueNppContext& nppCtx,
    at::cuda::CUDAStream nvdecStream,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  auto frameDims = FrameDims(avFrame->height, avFrame->width);
  torch::Tensor dst;
  if (preAllocatedOutputTensor.has_value()) {
    dst = preAllocatedOutputTensor.value();
  } else {
    dst = allocateEmptyHWCTensor(frameDims, device);
  }

  // We need to make sure NVDEC has finished decoding a frame before
  // color-converting it with NPP.
  // So we make the NPP stream wait for NVDEC to finish.
  at::cuda::CUDAStream nppStream =
      at::cuda::getCurrentCUDAStream(device.index());
  at::cuda::CUDAEvent nvdecDoneEvent;
  nvdecDoneEvent.record(nvdecStream);
  nvdecDoneEvent.block(nppStream);

  nppCtx->hStream = nppStream.stream();
  cudaError_t err = cudaStreamGetFlags(nppCtx->hStream, &nppCtx->nStreamFlags);
  TORCH_CHECK(
      err == cudaSuccess,
      "cudaStreamGetFlags failed: ",
      cudaGetErrorString(err));

  NppiSize oSizeROI = {frameDims.width, frameDims.height};
  Npp8u* yuvData[2] = {avFrame->data[0], avFrame->data[1]};

  NppStatus status;

  // For background, see
  // Note [YUV -> RGB Color Conversion, color space and color range]
  if (avFrame->colorspace == AVColorSpace::AVCOL_SPC_BT709) {
    if (avFrame->color_range == AVColorRange::AVCOL_RANGE_JPEG) {
      // NPP provides a pre-defined color conversion function for BT.709 full
      // range: nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx. But it's not closely
      // matching the results we have on CPU. So we're using a custom color
      // conversion matrix, which provides more accurate results. See the note
      // mentioned above for details, and headaches.

      int srcStep[2] = {avFrame->linesize[0], avFrame->linesize[1]};

      status = nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(
          yuvData,
          srcStep,
          static_cast<Npp8u*>(dst.data_ptr()),
          dst.stride(0),
          oSizeROI,
          bt709FullRangeColorTwist,
          *nppCtx);
    } else {
      // If not full range, we assume studio limited range.
      // The color conversion matrix for BT.709 limited range should be:
      // static const Npp32f bt709LimitedRangeColorTwist[3][4] = {
      //   {1.16438356f, 0.0f, 1.79274107f, -16.0f},
      //   {1.16438356f, -0.213248614f, -0.5329093290f, -128.0f},
      //   {1.16438356f, 2.11240179f, 0.0f, -128.0f}
      // };
      // We get very close results to CPU with that, but using the pre-defined
      // nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx seems to be even more accurate.
      status = nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(
          yuvData,
          avFrame->linesize[0],
          static_cast<Npp8u*>(dst.data_ptr()),
          dst.stride(0),
          oSizeROI,
          *nppCtx);
    }
  } else {
    // TODO we're assuming BT.601 color space (and probably limited range) by
    // calling nppiNV12ToRGB_8u_P2C3R_Ctx. We should handle BT.601 full range,
    // and other color-spaces like 2020.
    status = nppiNV12ToRGB_8u_P2C3R_Ctx(
        yuvData,
        avFrame->linesize[0],
        static_cast<Npp8u*>(dst.data_ptr()),
        dst.stride(0),
        oSizeROI,
        *nppCtx);
  }
  TORCH_CHECK(status == NPP_SUCCESS, "Failed to convert NV12 frame.");

  return dst;
}

UniqueNppContext getNppStreamContext(const torch::Device& device) {
  int deviceIndex = getDeviceIndex(device);

  UniqueNppContext nppCtx = g_cached_npp_ctxs.get(device);
  if (nppCtx) {
    return nppCtx;
  }

  // From 12.9, NPP recommends using a user-created NppStreamContext and using
  // the `_Ctx()` calls:
  // https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#npp-release-12-9-update-1
  // And the nppGetStreamContext() helper is deprecated. We are explicitly
  // supposed to create the NppStreamContext manually from the CUDA device
  // properties:
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/d97803a40fab83c058bb3d68b6c38bd6eebfff43/NPP/README.md?plain=1#L54-L72

  nppCtx = std::make_unique<NppStreamContext>();
  cudaDeviceProp prop{};
  cudaError_t err = cudaGetDeviceProperties(&prop, deviceIndex);
  TORCH_CHECK(
      err == cudaSuccess,
      "cudaGetDeviceProperties failed: ",
      cudaGetErrorString(err));

  nppCtx->nCudaDeviceId = deviceIndex;
  nppCtx->nMultiProcessorCount = prop.multiProcessorCount;
  nppCtx->nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  nppCtx->nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  nppCtx->nSharedMemPerBlock = prop.sharedMemPerBlock;
  nppCtx->nCudaDevAttrComputeCapabilityMajor = prop.major;
  nppCtx->nCudaDevAttrComputeCapabilityMinor = prop.minor;

  return nppCtx;
}

void returnNppStreamContextToCache(
    const torch::Device& device,
    UniqueNppContext nppCtx) {
  if (nppCtx) {
    g_cached_npp_ctxs.addIfCacheHasCapacity(device, std::move(nppCtx));
  }
}

void validatePreAllocatedTensorShape(
    const std::optional<torch::Tensor>& preAllocatedOutputTensor,
    const UniqueAVFrame& avFrame) {
  // Note that CUDA does not yet support transforms, so the only possible
  // frame dimensions are the raw decoded frame's dimensions.
  auto frameDims = FrameDims(avFrame->height, avFrame->width);

  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == frameDims.height) &&
            (shape[1] == frameDims.width) && (shape[2] == 3),
        "Expected tensor of shape ",
        frameDims.height,
        "x",
        frameDims.width,
        "x3, got ",
        shape);
  }
}

int getDeviceIndex(const torch::Device& device) {
  // PyTorch uses int8_t as its torch::DeviceIndex, but FFmpeg and CUDA
  // libraries use int. So we use int, too.
  int deviceIndex = static_cast<int>(device.index());
  TORCH_CHECK(
      deviceIndex >= -1 && deviceIndex < MAX_CUDA_GPUS,
      "Invalid device index = ",
      deviceIndex);

  if (deviceIndex == -1) {
    TORCH_CHECK(
        cudaGetDevice(&deviceIndex) == cudaSuccess,
        "Failed to get current CUDA device.");
  }
  return deviceIndex;
}

} // namespace facebook::torchcodec
