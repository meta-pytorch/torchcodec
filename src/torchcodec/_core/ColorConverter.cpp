// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ColorConverter.h"

#include "Frame.h"
#include "StreamOptions.h"

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

  // The converter is stream-agnostic: it derives everything it needs
  // (dimensions, pixel format, colorspace, and on CUDA the device-ness and bit
  // depth of the data) from each frame it's given. The DeviceInterface is the
  // vendor extension point, so conversion still goes through it in order to
  // stay third-party aware.
  device_interface_->initialize_color_conversion(options);
}

torch::stable::Tensor ColorConverter::convert(UniqueAVFrame& av_frame) {
  FrameOutput frame_output;
  device_interface_->convert_av_frame_to_frame_output(
      av_frame, frame_output, std::nullopt);
  return frame_output.data;
}

} // namespace facebook::torchcodec
