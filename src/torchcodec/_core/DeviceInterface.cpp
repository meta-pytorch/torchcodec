// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DeviceInterface.h"
#include <cstring>
#include <map>
#include <mutex>

namespace facebook::torchcodec {

namespace {
using DeviceInterfaceMap =
    std::map<DeviceInterfaceKey, CreateDeviceInterfaceFn>;
static std::mutex g_interface_mutex;

DeviceInterfaceMap& getDeviceMap() {
  static DeviceInterfaceMap deviceMap;
  return deviceMap;
}

std::string getDeviceTypeString(const std::string& device) {
  size_t pos = device.find(':');
  if (pos == std::string::npos) {
    return device;
  }
  return device.substr(0, pos);
}

// Parse device type from string (e.g., "cpu", "cuda")
// The stable ABI Device doesn't support string-based construction
StableDeviceType parseDeviceType(const std::string& deviceType) {
  if (deviceType == "cpu") {
    return kStableCPU;
  } else if (deviceType == "cuda") {
    return kStableCUDA;
  } else {
    STABLE_CHECK(false, "Unknown device type: ", deviceType);
  }
}

} // namespace

bool registerDeviceInterface(
    const DeviceInterfaceKey& key,
    CreateDeviceInterfaceFn createInterface) {
  std::scoped_lock lock(g_interface_mutex);
  DeviceInterfaceMap& deviceMap = getDeviceMap();

  STABLE_CHECK(
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
    const std::string device,
    const std::string variant) {
  std::scoped_lock lock(g_interface_mutex);
  std::string deviceType = getDeviceTypeString(device);

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

  STABLE_CHECK(
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
    const std::string_view variant) {
  DeviceInterfaceKey key(device.type(), variant);
  std::scoped_lock lock(g_interface_mutex);
  DeviceInterfaceMap& deviceMap = getDeviceMap();

  auto it = deviceMap.find(key);
  if (it != deviceMap.end()) {
    return std::unique_ptr<DeviceInterface>(it->second(device));
  }

  STABLE_CHECK(
      false,
      "No device interface found for device type: ",
      static_cast<int>(device.type()),
      " variant: '",
      variant,
      "'");
}

StableTensor rgbAVFrameToTensor(const UniqueAVFrame& avFrame) {
  STABLE_CHECK(
      avFrame->format == AV_PIX_FMT_RGB24,
      "Expected RGB24 format, got: ",
      avFrame->format);

  int height = avFrame->height;
  int width = avFrame->width;

  // Allocate output tensor
  // Note: Stable ABI's from_blob doesn't support custom deleters,
  // so we allocate and copy the data instead
  StableTensor result =
      stableEmpty({height, width, 3}, kStableUInt8, StableDevice(kStableCPU));

  uint8_t* dst = result.mutable_data_ptr<uint8_t>();
  const uint8_t* src = avFrame->data[0];
  int rowSize = width * 3;
  int linesize = avFrame->linesize[0];

  if (linesize == rowSize) {
    // Contiguous - single copy
    std::memcpy(dst, src, static_cast<size_t>(height) * rowSize);
  } else {
    // Non-contiguous - copy row by row
    for (int y = 0; y < height; y++) {
      std::memcpy(dst + y * rowSize, src + y * linesize, rowSize);
    }
  }

  return result;
}

} // namespace facebook::torchcodec
