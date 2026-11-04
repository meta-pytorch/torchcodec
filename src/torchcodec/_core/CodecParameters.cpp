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
  // From FFmpeg 6.1, this reads the display matrix from the coded side data,
  // which we just copied into codecpar. Before that, it reads it from the
  // AVStream, and display_matrix is the only place where we keep it: see
  // get_stream_side_data().
  if (const int32_t* matrix = get_display_matrix_from_stream(&av_stream)) {
    display_matrix.emplace();
    std::copy(matrix, matrix + display_matrix->size(), display_matrix->begin());
  }
}

std::optional<double> CodecParameters::get_rotation() const {
  return get_rotation_from_display_matrix(
      display_matrix.has_value() ? display_matrix->data() : nullptr);
}

std::vector<AVPacketSideData> CodecParameters::get_stream_side_data() const {
  std::vector<AVPacketSideData> entries;
#if FFMPEG_HAS_CODED_SIDE_DATA
  // The display matrix, if any, is one of them.
  for (int i = 0; i < codecpar->nb_coded_side_data; ++i) {
    entries.push_back(codecpar->coded_side_data[i]);
  }
#else
  // AVCodecParameters has no side data before FFmpeg 6.1: it was on the
  // AVStream, and the display matrix is the only one we keep.
  if (display_matrix.has_value()) {
    AVPacketSideData entry = {};
    entry.data = reinterpret_cast<uint8_t*>(
        const_cast<int32_t*>(display_matrix->data()));
    entry.size = sizeof(*display_matrix);
    entry.type = AV_PKT_DATA_DISPLAYMATRIX;
    entries.push_back(entry);
  }
#endif
  return entries;
}

} // namespace facebook::torchcodec
