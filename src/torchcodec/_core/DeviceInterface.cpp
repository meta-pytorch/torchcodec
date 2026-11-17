// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DeviceInterface.h"
#include <map>
#include <mutex>
#include "StableABICompat.h"

namespace facebook::torchcodec {

namespace {
using DeviceInterfaceMap =
    std::map<DeviceInterfaceKey, CreateDeviceInterfaceFn>;
static std::mutex g_interface_mutex;

DeviceInterfaceMap& get_device_map() {
  static DeviceInterfaceMap device_map;
  return device_map;
}

std::string get_device_type_string(const std::string& device) {
  size_t pos = device.find(':');
  if (pos == std::string::npos) {
    return device;
  }
  return device.substr(0, pos);
}

// Parse device type from string (e.g., "cpu", "cuda")
// TODO_STABLE_ABI: we might need to support more device types, i.e. those from
// https://github.com/pytorch/pytorch/blob/main/torch/headeronly/core/DeviceType.h
// Ideally we'd remove this helper?
StableDeviceType parse_device_type(const std::string& device_type) {
  if (device_type == "cpu") {
    return kStableCPU;
  } else if (device_type == "cuda") {
    return kStableCUDA;
  } else if (device_type == "xpu") {
    return kStableXPU;
  } else {
    STD_TORCH_CHECK(false, "Unknown device type: ", device_type);
  }
}

} // namespace

bool register_device_interface(
    const DeviceInterfaceKey& key,
    CreateDeviceInterfaceFn create_interface) {
  std::scoped_lock lock(g_interface_mutex);
  DeviceInterfaceMap& device_map = get_device_map();

  STD_TORCH_CHECK(
      device_map.find(key) == device_map.end(),
      "Device interface already registered for device type ",
      static_cast<int>(key.device_type),
      " variant '",
      key.variant,
      "'");
  device_map.insert({key, create_interface});

  return true;
}

void validate_device_interface(
    const std::string& device,
    const std::string& variant) {
  std::scoped_lock lock(g_interface_mutex);
  std::string device_type = get_device_type_string(device);

  DeviceInterfaceMap& device_map = get_device_map();

  // Find device interface that matches device type and variant
  StableDeviceType device_type_enum = parse_device_type(device_type);

  auto device_interface = std::find_if(
      device_map.begin(),
      device_map.end(),
      [&](const std::pair<DeviceInterfaceKey, CreateDeviceInterfaceFn>& arg) {
        return arg.first.device_type == device_type_enum &&
            arg.first.variant == variant;
      });

  STD_TORCH_CHECK(
      device_interface != device_map.end(),
      "Unsupported device: ",
      device,
      " (device type: ",
      device_type,
      ", variant: ",
      variant,
      ")");
}

std::unique_ptr<DeviceInterface> create_device_interface(
    const StableDevice& device,
    const std::string_view variant) {
  DeviceInterfaceKey key(device.type(), variant);
  std::scoped_lock lock(g_interface_mutex);
  DeviceInterfaceMap& device_map = get_device_map();

  auto it = device_map.find(key);
  if (it != device_map.end()) {
    return std::unique_ptr<DeviceInterface>(it->second(device));
  }

  STD_TORCH_CHECK(
      false,
      "No device interface found for device type: ",
      static_cast<int>(device.type()),
      " variant: '",
      variant,
      "'");
}

torch::stable::Tensor av_frame_to_hwc_tensor(const AVFrame& av_frame) {
  auto format = static_cast<AVPixelFormat>(av_frame.format);
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(format);
  STD_TORCH_CHECK(
      desc != nullptr, "Unknown pixel format: ", static_cast<int>(format));
  int64_t num_channels = desc->nb_components;
  int depth = desc->comp[0].depth;
  int bytes_per_element = depth > 8 ? 2 : 1;
  // Tensors hold 16-bit values in the host's byte order.
  bool is_big_endian = (desc->flags & AV_PIX_FMT_FLAG_BE) != 0;
  bool is_native_endian =
      bytes_per_element == 1 || is_big_endian == (AV_HAVE_BIGENDIAN != 0);
  STD_TORCH_CHECK(
      av_pix_fmt_count_planes(format) == 1 &&
          !(desc->flags &
            (AV_PIX_FMT_FLAG_PAL | AV_PIX_FMT_FLAG_BITSTREAM |
             AV_PIX_FMT_FLAG_HWACCEL)) &&
          (depth == 8 || depth == 16) &&
          desc->comp[0].step == num_channels * bytes_per_element &&
          is_native_endian,
      "Expected a packed, native-endian pixel format with 8-bit or 16-bit "
      "components, got ",
      desc->name);

  int height = av_frame.height;
  int width = av_frame.width;
  AVFrame* cloned_av_frame = av_frame_clone(&av_frame);

  auto deleter = [cloned_av_frame](void*) {
    UniqueAVFrame av_frame_to_delete(cloned_av_frame);
  };

  std::vector<int64_t> shape = {height, width, num_channels};

  // linesize is in bytes, but torch strides are in elements, so divide.
  auto dtype = bytes_per_element == 2 ? kStableUInt16 : kStableUInt8;
  std::vector<int64_t> strides = {
      cloned_av_frame->linesize[0] / bytes_per_element, num_channels, 1};

  return torch::stable::from_blob(
      cloned_av_frame->data[0],
      shape,
      strides,
      StableDevice(kStableCPU),
      dtype,
      deleter);
}

} // namespace facebook::torchcodec
