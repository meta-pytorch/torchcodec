// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "CodecParameters.h"

#include <algorithm>

namespace facebook::torchcodec {

CodecParameters get_codec_parameters(
    const AVStream& av_stream,
    const AVInputFormat* container_format) {
  CodecParameters params;
  params.codecpar.reset(avcodec_parameters_alloc());
  STD_TORCH_CHECK(
      params.codecpar != nullptr, "Failed to allocate AVCodecParameters.");
  int status =
      avcodec_parameters_copy(params.codecpar.get(), av_stream.codecpar);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to copy the codec parameters: ",
      get_ffmpeg_error_string_from_error_code(status));
  params.time_base = av_stream.time_base;
  params.r_frame_rate = av_stream.r_frame_rate;
  if (const int32_t* matrix = get_display_matrix_from_stream(&av_stream)) {
    params.display_matrix.emplace();
    std::copy(
        matrix,
        matrix + params.display_matrix->size(),
        params.display_matrix->begin());
  }
  params.container_format = container_format;
  return params;
}

std::optional<double> get_rotation(const CodecParameters& codec_parameters) {
  if (!codec_parameters.display_matrix.has_value()) {
    return std::nullopt;
  }
  return get_rotation_from_display_matrix(
      codec_parameters.display_matrix->data());
}

} // namespace facebook::torchcodec
