// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "FilterGraph.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"

extern "C" {
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

namespace facebook::torchcodec {

FiltersConfig::FiltersConfig(
    int inputWidth,
    int inputHeight,
    AVPixelFormat inputFormat,
    AVRational inputAspectRatio,
    int outputWidth,
    int outputHeight,
    AVPixelFormat outputFormat,
    const std::string& filtergraphStr,
    AVRational timeBase,
    AVBufferRef* hwFramesCtx)
    : inputWidth(inputWidth),
      inputHeight(inputHeight),
      inputFormat(inputFormat),
      inputAspectRatio(inputAspectRatio),
      outputWidth(outputWidth),
      outputHeight(outputHeight),
      outputFormat(outputFormat),
      filtergraphStr(filtergraphStr),
      timeBase(timeBase),
      hwFramesCtx(hwFramesCtx) {}

bool operator==(const AVRational& lhs, const AVRational& rhs) {
  return lhs.num == rhs.num && lhs.den == rhs.den;
}

bool FiltersConfig::operator==(const FiltersConfig& other) const {
  return inputWidth == other.inputWidth && inputHeight == other.inputHeight &&
      inputFormat == other.inputFormat && outputWidth == other.outputWidth &&
      outputHeight == other.outputHeight &&
      outputFormat == other.outputFormat &&
      filtergraphStr == other.filtergraphStr && timeBase == other.timeBase &&
      hwFramesCtx.get() == other.hwFramesCtx.get();
}

bool FiltersConfig::operator!=(const FiltersConfig& other) const {
  return !(*this == other);
}

FilterGraph::FilterGraph(
    const FiltersConfig& filtersConfig,
    const VideoStreamOptions& videoStreamOptions) {
  filterGraph_.reset(avfilter_graph_alloc());
  STD_TORCH_CHECK(
      filterGraph_.get() != nullptr, "Failed to allocate filter graph");

  if (videoStreamOptions.ffmpegThreadCount.has_value()) {
    filterGraph_->nb_threads = videoStreamOptions.ffmpegThreadCount.value();
  }

  // Configure the source context.
  const AVFilter* bufferSrc = avfilter_get_by_name("buffer");
  UniqueAVBufferSrcParameters srcParams(av_buffersrc_parameters_alloc());
  STD_TORCH_CHECK(srcParams, "Failed to allocate buffersrc params");

  srcParams->format = filtersConfig.inputFormat;
  srcParams->width = filtersConfig.inputWidth;
  srcParams->height = filtersConfig.inputHeight;
  srcParams->sample_aspect_ratio = filtersConfig.inputAspectRatio;
  srcParams->time_base = filtersConfig.timeBase;
  if (filtersConfig.hwFramesCtx) {
    srcParams->hw_frames_ctx = av_buffer_ref(filtersConfig.hwFramesCtx.get());
  }

  sourceContext_ =
      avfilter_graph_alloc_filter(filterGraph_.get(), bufferSrc, "in");
  STD_TORCH_CHECK(sourceContext_, "Failed to allocate filter graph");

  int status = av_buffersrc_parameters_set(sourceContext_, srcParams.get());
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to create filter graph: ",
      getFFMPEGErrorStringFromErrorCode(status));

  status = avfilter_init_str(sourceContext_, nullptr);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to create filter graph : ",
      getFFMPEGErrorStringFromErrorCode(status));

  // Configure the sink context.
  const AVFilter* bufferSink = avfilter_get_by_name("buffersink");
  STD_TORCH_CHECK(bufferSink != nullptr, "Failed to get buffersink filter.");

  sinkContext_ = createAVFilterContextWithOptions(
      filterGraph_.get(), bufferSink, filtersConfig.outputFormat);
  STD_TORCH_CHECK(
      sinkContext_ != nullptr, "Failed to create and configure buffersink");

  // Create the filtergraph nodes based on the source and sink contexts.
  UniqueAVFilterInOut outputs(avfilter_inout_alloc());
  outputs->name = av_strdup("in");
  outputs->filter_ctx = sourceContext_;
  outputs->pad_idx = 0;
  outputs->next = nullptr;

  UniqueAVFilterInOut inputs(avfilter_inout_alloc());
  inputs->name = av_strdup("out");
  inputs->filter_ctx = sinkContext_;
  inputs->pad_idx = 0;
  inputs->next = nullptr;

  // Create the filtergraph specified by the filtergraph string in the context
  // of the inputs and outputs. Note the dance we have to do with release and
  // resetting the output and input nodes because FFmpeg modifies them in place.
  AVFilterInOut* outputsTmp = outputs.release();
  AVFilterInOut* inputsTmp = inputs.release();
  status = avfilter_graph_parse_ptr(
      filterGraph_.get(),
      filtersConfig.filtergraphStr.c_str(),
      &inputsTmp,
      &outputsTmp,
      nullptr);
  outputs.reset(outputsTmp);
  inputs.reset(inputsTmp);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to parse filter description: ",
      getFFMPEGErrorStringFromErrorCode(status),
      ", provided filters: " + filtersConfig.filtergraphStr);

  // Check filtergraph validity and configure links and formats.
  status = avfilter_graph_config(filterGraph_.get(), nullptr);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to configure filter graph: ",
      getFFMPEGErrorStringFromErrorCode(status),
      ", provided filters: " + filtersConfig.filtergraphStr);
}

UniqueAVFrame FilterGraph::convert(const UniqueAVFrame& avFrame) {
  int status = av_buffersrc_write_frame(sourceContext_, avFrame.get());
  STD_TORCH_CHECK(
      status >= AVSUCCESS, "Failed to add frame to buffer source context");

  UniqueAVFrame filteredAVFrame(av_frame_alloc());
  status = av_buffersink_get_frame(sinkContext_, filteredAVFrame.get());
  STD_TORCH_CHECK(
      status >= AVSUCCESS, "Failed to get frame from buffer sink context");

  return filteredAVFrame;
}

} // namespace facebook::torchcodec
