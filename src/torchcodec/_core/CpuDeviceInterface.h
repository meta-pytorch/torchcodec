// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "FilterGraph.h"
#include "SwScale.h"

namespace facebook::torchcodec {

class CpuDeviceInterface : public DeviceInterface {
 public:
  CpuDeviceInterface(const StableDevice& device);

  virtual ~CpuDeviceInterface() {}

  std::optional<const AVCodec*> find_codec(
      [[maybe_unused]] const AVCodecID& codec_id,
      [[maybe_unused]] bool is_decoder = true) override {
    return std::nullopt;
  }

  virtual void initialize(const SharedAVCodecContext& codec_context) override;

  virtual void initialize_video_decoding(
      const AVStream* av_stream,
      const UniqueDecodingAVFormatContext& av_format_ctx,
      const VideoStreamOptions& video_stream_options) override;

  virtual void initialize_color_conversion(
      const VideoStreamOptions& video_stream_options,
      const std::vector<std::unique_ptr<Transform>>& transforms,
      const std::optional<FrameDims>& resized_output_dims) override;

  virtual void initialize_audio(
      const AudioStreamOptions& audio_stream_options,
      double stream_start_seconds = 0.0) override;

  virtual std::optional<torch::stable::Tensor> maybe_flush_audio_buffers()
      override;

  void flush() override;

  void convert_av_frame_to_frame_output(
      const AVFrame& av_frame,
      FrameOutput& frame_output,
      std::optional<torch::stable::Tensor> pre_allocated_output_tensor)
      override;

  UniqueAVFrame convert_tensor_to_av_frame_for_encoding(
      const torch::stable::Tensor& tensor,
      int frame_index,
      AVCodecContext* codec_context) override;

  AVPixelFormat get_encoding_pixel_format(
      const AVCodec& av_codec,
      const std::optional<std::string>& user_pixel_format) const override;

  std::string get_details() override;

 private:
  void convert_audio_av_frame_to_frame_output(
      const AVFrame& src_av_frame,
      FrameOutput& frame_output);

  void convert_video_av_frame_to_frame_output(
      const AVFrame& av_frame,
      FrameOutput& frame_output,
      std::optional<torch::stable::Tensor> pre_allocated_output_tensor);

  torch::stable::Tensor convert_av_frame_to_tensor_using_filter_graph(
      const AVFrame& av_frame,
      const FrameDims& output_dims);

  ColorConversionLibrary get_color_conversion_library(
      const FrameDims& input_dims,
      const FrameDims& output_dims) const;

  VideoStreamOptions video_stream_options_;
  // Default used when color conversion runs standalone (no stream to derive it
  // from, e.g. the ColorConverter block API). initialize_video_decoding()
  // overrides it from the stream when there is one. Its value doesn't matter on
  // color-conversion-only mode, but filtergraph still expects it.
  AVRational time_base_ = {1, AV_TIME_BASE};
  AVPixelFormat output_pixel_format_;

  // If the resized output dimensions are present, then we always use those as
  // the output frame's dimensions. If they are not present, then we use the
  // dimensions of the raw decoded frame. Note that we do not know the
  // dimensions of the raw decoded frame until very late; we learn it in
  // convertAVFrameToFrameOutput(). Deciding the final output frame's actual
  // dimensions late allows us to handle video streams with variable
  // resolutions.
  std::optional<FrameDims> resized_output_dims_;

  // Color-conversion objects. Only one of filterGraph_ and swScale_ should
  // be non-null. Which one we use is determined dynamically in
  // getColorConversionLibrary() each time we decode a frame.
  //
  // Creating both filterGraph_ and swScale_ is relatively expensive, so we
  // reuse them across frames. However, it is possible that subsequent frames
  // are different enough (change in dimensions) that we can't reuse the color
  // conversion object. We store the relevant frame config from the frame used
  // to create the object last time. We always compare the current frame's info
  // against the previous one to determine if we need to recreate the color
  // conversion object.
  std::unique_ptr<FilterGraph> filter_graph_;
  FiltersConfig prev_filters_config_;
  std::unique_ptr<SwScale> sw_scale_;

  // Cached swscale context for encoding (tensor -> AVFrame pixel format
  // conversion).
  UniqueSwsContext encoding_sws_context_;

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
  bool are_transforms_sw_scale_compatible_;
  bool user_requested_sw_scale_;

  // The flags we supply to the resize swscale context. The flags control the
  // resizing algorithm. We default to bilinear. Users can override this with a
  // ResizeTransform that specifies a different interpolation mode.
  int sws_flags_ = SWS_BILINEAR;

  bool initialized_ = false;

  // Audio-specific members
  AudioStreamOptions audio_stream_options_;
  UniqueSwrContext swr_context_;

  // Input samples still to be discarded before feeding swresample, so that the
  // first sample it is fed sits on the stream's output sample grid. Always
  // less than src_rate / gcd(rates), and can span several frames when a frame
  // holds fewer samples than that. See [Resampler Grid Alignment].
  int64_t swr_input_samples_to_skip_ = 0;

  // Start of the stream, which the output sample grid is anchored on. Streams
  // don't necessarily start at 0.
  double audio_stream_start_seconds_ = 0.0;
};

} // namespace facebook::torchcodec
