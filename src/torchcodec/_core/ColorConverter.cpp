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

namespace {
// Only ever used to name a device in the error message below.
std::string printable(const StableDevice& device) {
  std::string name = device_type_name(device.type());
  if (device.type() != kStableCPU && device.index() >= 0) {
    name += ":" + std::to_string(device.index());
  }
  return name;
}
} // namespace

ColorConverter::ColorConverter(
    const StableDevice& device,
    OutputDtypeConfig output_dtype_config)
    : device_(device), output_dtype_config_(output_dtype_config) {
  device_interface_ = create_device_interface(device);
  STD_TORCH_CHECK(
      device_interface_ != nullptr,
      "Failed to create device interface. This should never happen, please report.");
  device_ = device_interface_->device(); // resolved, so we don't have to
}

void ColorConverter::maybe_initialize_interface(OutputDtype output_dtype) {
  // Interface initialization is done per-frame, not in the constructor: with
  // AUTO, the desired output dtype is only known once we see a frame, and it
  // can differ from one frame to the next.
  if (initialized_output_dtype_.has_value() &&
      *initialized_output_dtype_ == output_dtype) {
    return;
  }

  VideoStreamOptions options;
  options.device = device_;

  std::vector<std::unique_ptr<Transform>> no_transforms;
  device_interface_->initialize_color_conversion(
      output_dtype,
      options,
      no_transforms,
      /*resized_output_dims=*/std::nullopt);
  initialized_output_dtype_ = output_dtype;
}

torch::stable::Tensor ColorConverter::convert(
    const AVFrame& av_frame,
    const StableDevice& frame_device) {
  // TODO_API_BREAKDOWN CC P2: OK, it's not fantastic that we have to pass the
  // frame's device. Especially given the related design TODO about whether the
  // RawFrame should carry that device field at all. Maybe it should, maybe it's
  // overkill. I think the main alternative is to retrieve the device from the
  // AVFrame, it's possible, but likely requires moving the
  // StandaloneFrameAttachedData to the public header.
  STD_TORCH_CHECK(
      frame_device == device_,
      "This ColorConverter is on ",
      printable(device_),
      " but the frame's samples are on ",
      printable(frame_device),
      ". A ColorConverter only converts frames that are already on its own "
      "device: create one per device, or move the RGB output afterwards.");

  OutputDtype output_dtype = resolve_output_dtype(
      output_dtype_config_, static_cast<AVPixelFormat>(av_frame.format));
  maybe_initialize_interface(output_dtype);

  FrameOutput frame_output;
  device_interface_->convert_av_frame_to_frame_output(
      av_frame, frame_output, std::nullopt);

  // TODO_API_BREAKDOWN PERF P2: on CPU this is a lot slower than the filter
  // graph's transpose.
  frame_output.data = rotate_hwc_tensor(
      frame_output.data,
      rotation_from_degrees(get_rotation_from_frame(av_frame)));

  return convert_to_output_dtype(frame_output.data, output_dtype);
}

} // namespace facebook::torchcodec
