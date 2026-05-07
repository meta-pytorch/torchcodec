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

DeviceInterfaceMap& getDeviceMap() {
  static DeviceInterfaceMap deviceMap;
  return deviceMap;
}

std::string_view getDeviceTypeString(std::string_view device) {
  size_t pos = device.find(':');
  if (pos == std::string_view::npos) {
    return device;
  }
  return device.substr(0, pos);
}

// Parse device type from string (e.g., "cpu", "cuda")
// TODO_STABLE_ABI: we might need to support more device types, i.e. those from
// https://github.com/pytorch/pytorch/blob/main/torch/headeronly/core/DeviceType.h
// Ideally we'd remove this helper?
StableDeviceType parseDeviceType(std::string_view deviceType) {
  if (deviceType == "cpu") {
    return kStableCPU;
  } else if (deviceType == "cuda") {
    return kStableCUDA;
  } else if (deviceType == "xpu") {
    return kStableXPU;
  } else {
    STD_TORCH_CHECK(false, "Unknown device type: ", deviceType);
  }
}

} // namespace

bool registerDeviceInterface(
    const DeviceInterfaceKey& key,
    CreateDeviceInterfaceFn createInterface) {
  std::scoped_lock lock(g_interface_mutex);
  DeviceInterfaceMap& deviceMap = getDeviceMap();

  STD_TORCH_CHECK(
      deviceMap.find(key) == deviceMap.end(),
      "Device interface already registered for device type ",
      static_cast<int>(key.deviceType),
      " variant '",
      key.variant,
      "'");
  deviceMap.insert({key, createInterface});

  return true;
}

void validateDeviceInterface(
    std::string_view device,
    std::string_view variant) {
  std::scoped_lock lock(g_interface_mutex);
  std::string_view deviceType = getDeviceTypeString(device);

  DeviceInterfaceMap& deviceMap = getDeviceMap();

  // Find device interface that matches device type and variant
  StableDeviceType deviceTypeEnum = parseDeviceType(deviceType);

  auto deviceInterface = std::find_if(
      deviceMap.begin(),
      deviceMap.end(),
      [&](const std::pair<DeviceInterfaceKey, CreateDeviceInterfaceFn>& arg) {
        return arg.first.deviceType == deviceTypeEnum &&
            arg.first.variant == variant;
      });

  STD_TORCH_CHECK(
      deviceInterface != deviceMap.end(),
      "Unsupported device: ",
      device,
      " (device type: ",
      deviceType,
      ", variant: ",
      variant,
      ")");
}

std::unique_ptr<DeviceInterface> createDeviceInterface(
    const StableDevice& device,
    std::string_view variant) {
  DeviceInterfaceKey key(device.type(), variant);
  std::scoped_lock lock(g_interface_mutex);
  DeviceInterfaceMap& deviceMap = getDeviceMap();

  auto it = deviceMap.find(key);
  if (it != deviceMap.end()) {
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

torch::stable::Tensor rgbAVFrameToTensor(const UniqueAVFrame& avFrame) {
  STD_TORCH_CHECK(avFrame->format == AV_PIX_FMT_RGB24, "Expected RGB24 format");

  int height = avFrame->height;
  int width = avFrame->width;
  std::vector<int64_t> shape = {height, width, 3};
  std::vector<int64_t> strides = {avFrame->linesize[0], 3, 1};
  AVFrame* avFrameClone = av_frame_clone(avFrame.get());

  auto deleter = [avFrameClone](void*) {
    UniqueAVFrame avFrameToDelete(avFrameClone);
  };

  return torch::stable::from_blob(
      avFrameClone->data[0],
      shape,
      strides,
      StableDevice(kStableCPU),
      kStableUInt8,
      deleter);
}

} // namespace facebook::torchcodec
