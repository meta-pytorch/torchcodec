// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/FilterGraph.h"

namespace facebook::torchcodec {

class CpuDeviceInterface : public DeviceInterface {
 public:
  CpuDeviceInterface(const torch::Device& device);

  virtual ~CpuDeviceInterface() {}

  std::optional<const AVCodec*> findCodec(
      [[maybe_unused]] const AVCodecID& codecId) override {
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

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

  std::string getDetails() override;

 private:
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

  // The filter we supply to filterGraph_, if it is used. The default is the
  // copy filter, which just copies the input to the output. Computationally, it
  // should be a no-op. If we get no user-provided transforms, we will use the
  // copy filter. Otherwise, we will construct the string from the transforms.
  //
  // Note that even if we only use the copy filter, we still get the desired
  // colorspace conversion. We construct the filtergraph with its output sink
  // set to RGB24.
  std::string filters_ = "copy";

  // The flags we supply to swsContext_, if it used. The flags control the
  // resizing algorithm. We default to bilinear. Users can override this with a
  // ResizeTransform.
  int swsFlags_ = SWS_BILINEAR;

  // Values set during initialization and referred to in
  // getColorConversionLibrary().
  bool areTransformsSwScaleCompatible_;
  bool userRequestedSwScale_;

  bool initialized_ = false;
};

} // namespace facebook::torchcodec
