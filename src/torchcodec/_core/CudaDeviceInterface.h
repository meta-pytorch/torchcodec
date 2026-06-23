// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "CUDACommon.h"
#include "DeviceInterface.h"
#include "FilterGraph.h"
#include "color_conversion.h"

namespace facebook::torchcodec {

class CudaDeviceInterface : public DeviceInterface {
 public:
  CudaDeviceInterface(const StableDevice& device);

  virtual ~CudaDeviceInterface();

  std::optional<const AVCodec*> find_codec(
      const AVCodecID& codec_id,
      bool is_decoder = true) override;

  void initialize(const SharedAVCodecContext& codec_context) override;

  void initialize_video(
      const AVStream* av_stream,
      const UniqueDecodingAVFormatContext& av_format_ctx,
      const VideoStreamOptions& video_stream_options,
      [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>&
          transforms,
      [[maybe_unused]] const std::optional<FrameDims>& resized_output_dims)
      override;

  void register_hardware_device_with_codec(
      AVCodecContext* codec_context) override;

  void convert_av_frame_to_frame_output(
      UniqueAVFrame& av_frame,
      FrameOutput& frame_output,
      std::optional<torch::stable::Tensor> pre_allocated_output_tensor)
      override;

  std::string get_details() override;

  UniqueAVFrame convert_tensor_to_av_frame_for_encoding(
      const torch::stable::Tensor& tensor,
      int frame_index,
      AVCodecContext* codec_context) override;

  void setup_hardware_frame_context_for_encoding(
      AVCodecContext* codec_context) override;

 private:
  // Our CUDA decoding code assumes NV12 format. In order to handle other
  // kinds of input, we need to convert them to NV12. Our current implementation
  // does this using filtergraph.
  UniqueAVFrame maybe_convert_av_frame_to_nv12_or_rgb24(
      UniqueAVFrame& av_frame);

  // We sometimes encounter frames that cannot be decoded on the CUDA device.
  // Rather than erroring out, we decode them on the CPU.
  std::unique_ptr<DeviceInterface> cpu_interface_;

  VideoStreamOptions video_stream_options_;
  AVRational time_base_;

  UniqueAVBufferRef hardware_device_ctx_;

  // This filtergraph instance is only used for NV12 format conversion in
  // maybeConvertAVFrameToNV12().
  std::unique_ptr<FiltersConfig> nv12_conversion_config_;
  std::unique_ptr<FilterGraph> nv12_conversion_;

  bool using_cpu_fallback_ = false;
  bool has_decoded_frame_ = false;

  CachedColorMatrix cached_color_matrix_;
};

} // namespace facebook::torchcodec
