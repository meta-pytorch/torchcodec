// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ColorConverter.h"

#include <optional>
#include <vector>

#include "Frame.h"
#include "StreamOptions.h"
#include "Transform.h"

namespace facebook::torchcodec {

ColorConverter::ColorConverter(
    const StableDevice& device,
    std::string_view device_variant) {
  device_interface_ = create_device_interface(device, device_variant);
  STD_TORCH_CHECK(
      device_interface_ != nullptr,
      "Failed to create device interface. This should never happen, please report.");

  VideoStreamOptions options;
  options.output_dtype = OutputDtype::UINT8; // dtype not exposed yet
  options.device = device;

  // No user transforms and no stream: the converter is stream-agnostic and
  // derives everything it needs from each frame.
  //
  // TODO_API_BREAKDOWN Need to refac/rethink all this. It seems unnatural that
  // the color-converter needs its own device_interface_, but at the same time
  // the color-conversion *must* be third-party aware, and the only way to
  // achieve that for now is via the interface.
  // This will become very relevant when we tackle CUDA, so we can defer until
  // then. For now this is an OK hack.
  std::vector<std::unique_ptr<Transform>> no_transforms;
  device_interface_->initialize_video(
      /*av_stream=*/nullptr,
      UniqueDecodingAVFormatContext{},
      options,
      no_transforms,
      /*resized_output_dims=*/std::nullopt);
}

torch::stable::Tensor ColorConverter::convert(UniqueAVFrame& av_frame) {
  FrameOutput frame_output;
  device_interface_->convert_av_frame_to_frame_output(
      av_frame, frame_output, std::nullopt);
  return frame_output.data;
}

} // namespace facebook::torchcodec
