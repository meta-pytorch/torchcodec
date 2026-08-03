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

  // TODO_API_BREAKDOWN It seems unnatural that the color-converter needs its
  // own device_interface_, but at the same time the color-conversion *must* be
  // third-party aware, and the only way to achieve that for now is via the
  // interface. Should at the very least write a note about this design that now
  // the DeviceInterface has different modes: decode only, color-convert only,
  // and decode+color-convert (which used to be the only mode).

  // TODO_API_BREAKDOWN: we shouldn't call initialize_video here, this is for
  // the decoding+color-convert mode. We should do something cleaner e.g.
  // initialize_color_convertion_only()
  std::vector<std::unique_ptr<Transform>> no_transforms;
  device_interface_->initialize_video(
      /*av_stream=*/nullptr,
      UniqueDecodingAVFormatContext{},
      options,
      no_transforms,
      /*resized_output_dims=*/std::nullopt);
}

torch::stable::Tensor ColorConverter::convert(const AVFrame& av_frame) {
  FrameOutput frame_output;
  device_interface_->convert_av_frame_to_frame_output(
      av_frame, frame_output, std::nullopt);
  return frame_output.data;
}

} // namespace facebook::torchcodec
