// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "FFMPEGCommon.h"
#include "StreamOptions.h"

namespace facebook::torchcodec {

struct FiltersConfig {
  int input_width = 0;
  int input_height = 0;
  AVPixelFormat input_format = AV_PIX_FMT_NONE;
  AVRational input_aspect_ratio = {0, 0};
  int output_width = 0;
  int output_height = 0;
  AVPixelFormat output_format = AV_PIX_FMT_NONE;
  std::string filtergraph_str;
  AVRational time_base = {0, 0};
  UniqueAVBufferRef hw_frames_ctx;

  FiltersConfig() = default;
  FiltersConfig(FiltersConfig&&) = default;
  FiltersConfig& operator=(FiltersConfig&&) = default;
  FiltersConfig(
      int input_width,
      int input_height,
      AVPixelFormat input_format,
      AVRational input_aspect_ratio,
      int output_width,
      int output_height,
      AVPixelFormat output_format,
      const std::string& filtergraph_str,
      AVRational time_base,
      AVBufferRef* hw_frames_ctx = nullptr);

  bool operator==(const FiltersConfig&) const;
  bool operator!=(const FiltersConfig&) const;
};

class FilterGraph {
 public:
  FilterGraph(
      const FiltersConfig& filters_config,
      const VideoStreamOptions& video_stream_options);

  UniqueAVFrame convert(const UniqueAVFrame& av_frame);

 private:
  UniqueAVFilterGraph filter_graph_;
  AVFilterContext* source_context_ = nullptr;
  AVFilterContext* sink_context_ = nullptr;
};

} // namespace facebook::torchcodec
