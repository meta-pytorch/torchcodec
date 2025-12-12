// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "FilterGraph.h"

namespace facebook::torchcodec {

class CpuDeviceInterface : public DeviceInterface {
 public:
  CpuDeviceInterface(const torch::Device& device);

  virtual ~CpuDeviceInterface() {}

  std::optional<const AVCodec*> findCodec(
      [[maybe_unused]] const AVCodecID& codecId,
      [[maybe_unused]] bool isDecoder = true) override {
    return std::nullopt;
  }

  virtual void initialize(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx,
      const SharedAVCodecContext& codecContext) override;

  virtual void initializeVideo(
      const VideoStreamOptions& videoStreamOptions,
      const std::vector<std::unique_ptr<Transform>>& transforms,
      const std::optional<FrameDims>& resizedOutputDims) override;

  virtual void initializeAudio(
      const AudioStreamOptions& audioStreamOptions) override;

  virtual std::optional<torch::Tensor> maybeFlushAudioBuffers() override;

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor) override;

  std::string getDetails() override;

 private:
  void convertAudioAVFrameToFrameOutput(
      UniqueAVFrame& srcAVFrame,
      FrameOutput& frameOutput);

  void convertVideoAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor);

  int convertAVFrameToTensorUsingSwScale(
      const UniqueAVFrame& avFrame,
      torch::Tensor& outputTensor,
      const FrameDims& outputDims);

  torch::Tensor convertAVFrameToTensorUsingFilterGraph(
      const UniqueAVFrame& avFrame,
      const FrameDims& outputDims);

  ColorConversionLibrary getColorConversionLibrary(
      const FrameDims& inputFrameDims) const;

  VideoStreamOptions videoStreamOptions_;
  AVRational timeBase_;

  // If the resized output dimensions are present, then we always use those as
  // the output frame's dimensions. If they are not present, then we use the
  // dimensions of the raw decoded frame. Note that we do not know the
  // dimensions of the raw decoded frame until very late; we learn it in
  // convertAVFrameToFrameOutput(). Deciding the final output frame's actual
  // dimensions late allows us to handle video streams with variable
  // resolutions.
  std::optional<FrameDims> resizedOutputDims_;

  // Color-conversion objects. Only one of filterGraph_ and swsContext_ should
  // be non-null. Which one we use is determined dynamically in
  // getColorConversionLibrary() each time we decode a frame.
  //
  // Creating both filterGraph_ and swsContext_ is relatively expensive, so we
  // reuse them across frames. However, it is possbile that subsequent frames
  // are different enough (change in dimensions) that we can't reuse the color
  // conversion object. We store the relevant frame context from the frame used
  // to create the object last time. We always compare the current frame's info
  // against the previous one to determine if we need to recreate the color
  // conversion object.
  //
  // TODO: The names of these fields is confusing, as the actual color
  //       conversion object for Sws has "context" in the name,  and we use
  //       "context" for the structs we store to know if we need to recreate a
  //       color conversion object. We should clean that up.
  std::unique_ptr<FilterGraph> filterGraph_;
  FiltersContext prevFiltersContext_;
  UniqueSwsContext swsContext_;
  SwsFrameContext prevSwsFrameContext_;

  // We pass these filters to FFmpeg's filtergraph API. It is a simple pipeline
  // of what FFmpeg calls "filters" to apply to decoded frames before returning
  // them. In the PyTorch ecosystem, we call these "transforms". During
  // initialization, we convert the user-supplied transforms into this string of
  // filters.
  //
  // Note that if there are no user-supplied transforms, then the default filter
  // we use is the copy filter, which is just an identity: it emits the output
  // frame unchanged. We supply such a filter because we can't supply just the
  // empty-string; we must supply SOME filter.
  //
  // See also [Tranform and Format Conversion Order] for more on filters.
  std::string filters_ = "copy";

  // Values set during initialization and referred to in
  // getColorConversionLibrary().
  bool areTransformsSwScaleCompatible_;
  bool userRequestedSwScale_;

  bool initialized_ = false;

  // Audio-specific members
  AudioStreamOptions audioStreamOptions_;
  UniqueSwrContext swrContext_;
};

} // namespace facebook::torchcodec
