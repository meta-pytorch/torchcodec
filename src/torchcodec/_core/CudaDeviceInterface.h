// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/CUDACommon.h"
#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FilterGraph.h"

namespace facebook::torchcodec {

class CudaDeviceInterface : public DeviceInterface {
 public:
  CudaDeviceInterface(const torch::Device& device);

  virtual ~CudaDeviceInterface();

  std::optional<const AVCodec*> findCodec(const AVCodecID& codecId) override;

  void initialize(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx,
      const SharedAVCodecContext& codecContext) override;

  void initializeVideo(
      const VideoStreamOptions& videoStreamOptions,
      [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>&
          transforms,
      [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims)
      override;

  void registerHardwareDeviceWithCodec(AVCodecContext* codecContext) override;

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

  std::string getDetails() override;

 private:
  // Our CUDA decoding code assumes NV12 format. In order to handle other
  // kinds of input, we need to convert them to NV12. Our current implementation
  // does this using filtergraph.
  UniqueAVFrame maybeConvertAVFrameToNV12OrRGB24(UniqueAVFrame& avFrame);

  // We sometimes encounter frames that cannot be decoded on the CUDA device.
  // Rather than erroring out, we decode them on the CPU.
  std::unique_ptr<DeviceInterface> cpuInterface_;

  VideoStreamOptions videoStreamOptions_;
  AVRational timeBase_;

  UniqueAVBufferRef hardwareDeviceCtx_;
  UniqueNppContext nppCtx_;

  // This filtergraph instance is only used for NV12 format conversion in
  // maybeConvertAVFrameToNV12().
  std::unique_ptr<FiltersContext> nv12ConversionContext_;
  std::unique_ptr<FilterGraph> nv12Conversion_;

  bool usingCPUFallback_ = false;
};

} // namespace facebook::torchcodec
