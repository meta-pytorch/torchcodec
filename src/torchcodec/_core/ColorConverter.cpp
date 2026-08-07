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

// Mirrors SingleStreamDecoder::maybe_permute_and_convert_to_float32().
torch::stable::Tensor maybe_convert_to_float32(
    torch::stable::Tensor& data,
    OutputDtype output_dtype) {
  if (output_dtype != OutputDtype::FLOAT32) {
    return data;
  }
  bool is_uint16 = data.scalar_type() == kStableUInt16;
  double max_val = is_uint16 ? 65535.0 : 255.0;
  auto as_float = torch::stable::to(data, kStableFloat32);
  return stable_div(as_float, max_val);
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

  // AUTO needs a frame to resolve; the first one may re-initialize us.
  initialize_for_output_dtype(
      output_dtype_config == OutputDtypeConfig::FLOAT32 ? OutputDtype::FLOAT32
                                                        : OutputDtype::UINT8);
}

void ColorConverter::initialize_for_output_dtype(OutputDtype output_dtype) {
  if (initialized_output_dtype_.has_value() &&
      *initialized_output_dtype_ == output_dtype) {
    return;
  }

  VideoStreamOptions options;
  options.output_dtype = output_dtype;
  options.device = device_;

  // TODO_API_BREAKDOWN P1 It seems unnatural that the color-converter needs its
  // own device_interface_, but at the same time the color-conversion *must* be
  // third-party aware, and the only way to achieve that for now is via the
  // interface.
  std::vector<std::unique_ptr<Transform>> no_transforms;
  device_interface_->initialize_color_conversion(
      options, no_transforms, /*resized_output_dims=*/std::nullopt);
  initialized_output_dtype_ = output_dtype;
}

OutputDtype ColorConverter::resolve_output_dtype(
    const AVFrame& av_frame) const {
  switch (output_dtype_config_) {
    case OutputDtypeConfig::UINT8:
      return OutputDtype::UINT8;
    case OutputDtypeConfig::FLOAT32:
      return OutputDtype::FLOAT32;
    case OutputDtypeConfig::AUTO: {
      const AVPixFmtDescriptor* desc =
          av_pix_fmt_desc_get(static_cast<AVPixelFormat>(av_frame.format));
      return (desc != nullptr && desc->comp[0].depth > 8) ? OutputDtype::FLOAT32
                                                          : OutputDtype::UINT8;
    }
  }
  return OutputDtype::UINT8;
}

torch::stable::Tensor ColorConverter::convert(const AVFrame& av_frame) {
  OutputDtype output_dtype = resolve_output_dtype(av_frame);
  initialize_for_output_dtype(output_dtype);

  FrameOutput frame_output;
  device_interface_->convert_av_frame_to_frame_output(
      av_frame, frame_output, std::nullopt);
  return maybe_convert_to_float32(frame_output.data, output_dtype);
}

} // namespace facebook::torchcodec
