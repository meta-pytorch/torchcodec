// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "CodecParameters.h"

#include <algorithm>

namespace facebook::torchcodec {

CodecParameters::CodecParameters(
    const AVStream& av_stream,
    const AVInputFormat* container_format)
    : codecpar(avcodec_parameters_alloc()),
      time_base(av_stream.time_base),
      r_frame_rate(av_stream.r_frame_rate),
      container_format(container_format) {
  STD_TORCH_CHECK(codecpar != nullptr, "Failed to allocate AVCodecParameters.");
  int status = avcodec_parameters_copy(codecpar.get(), av_stream.codecpar);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to copy the codec parameters: ",
      get_ffmpeg_error_string_from_error_code(status));
  if (const int32_t* matrix = get_display_matrix_from_stream(&av_stream)) {
    display_matrix.emplace();
    std::copy(matrix, matrix + display_matrix->size(), display_matrix->begin());
  }
}

std::optional<double> get_rotation(const CodecParameters& codec_parameters) {
  if (!codec_parameters.display_matrix.has_value()) {
    return std::nullopt;
  }
  return get_rotation_from_display_matrix(
      codec_parameters.display_matrix->data());
}

std::vector<AVPacketSideData> get_stream_side_data(
    const CodecParameters& codec_parameters) {
  std::vector<AVPacketSideData> entries;
  bool has_display_matrix = false;
  for (const AVPacketSideData* side_data :
       get_coded_side_data(*codec_parameters.codecpar)) {
    entries.push_back(*side_data);
    has_display_matrix |= side_data->type == AV_PKT_DATA_DISPLAYMATRIX;
  }
  if (codec_parameters.display_matrix.has_value() && !has_display_matrix) {
    AVPacketSideData entry = {};
    entry.data = reinterpret_cast<uint8_t*>(
        const_cast<int32_t*>(codec_parameters.display_matrix->data()));
    entry.size = sizeof(*codec_parameters.display_matrix);
    entry.type = AV_PKT_DATA_DISPLAYMATRIX;
    entries.push_back(entry);
  }
  return entries;
}

} // namespace facebook::torchcodec
