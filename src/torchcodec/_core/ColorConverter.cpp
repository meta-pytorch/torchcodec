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
    OutputDtypeConfig output_dtype_config)
    : device_(device), output_dtype_config_(output_dtype_config) {
  device_interface_ = create_device_interface(device);
  STD_TORCH_CHECK(
      device_interface_ != nullptr,
      "Failed to create device interface. This should never happen, please report.");
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
  options.output_dtype = output_dtype;
  options.device = device_;

  std::vector<std::unique_ptr<Transform>> no_transforms;
  device_interface_->initialize_color_conversion(
      options, no_transforms, /*resized_output_dims=*/std::nullopt);
  initialized_output_dtype_ = output_dtype;
}

torch::stable::Tensor ColorConverter::convert(const AVFrame& av_frame) {
  OutputDtype output_dtype = resolve_output_dtype(
      output_dtype_config_, static_cast<AVPixelFormat>(av_frame.format));
  maybe_initialize_interface(output_dtype);

  FrameOutput frame_output;
  device_interface_->convert_av_frame_to_frame_output(
      av_frame, frame_output, std::nullopt);

  // The SingleStreamDecoder rotates through the transform pipeline, which we
  // can't reuse: being unbound, we have no fixed input dims to build a
  // RotationTransform from. Rotating the converted RGB frame yields the same
  // pixels, because the CPU filter graph also applies its transforms after the
  // format conversion, see Note [Transform and Format Conversion Order].
  //
  // TODO_API_BREAKDOWN PERF P2: on CPU this is a lot slower than the filter
  // graph's transpose (roughly 3x at 480x270, and 1.5-3ms/frame at 720p):
  // rotating an HWC uint8 tensor is a strided gather with 3-byte inner runs,
  // where FFmpeg's transpose is SIMD and blocked. Only rotated videos pay it,
  // and it's ~20us/frame on CUDA, but a CPU pipeline decoding rotated videos
  // is measurably worse off than the VideoDecoder.
  frame_output.data = rotate_hwc_tensor(
      frame_output.data,
      rotation_from_degrees(get_rotation_from_frame(av_frame)));

  return convert_to_output_dtype(frame_output.data, output_dtype);
}

} // namespace facebook::torchcodec
