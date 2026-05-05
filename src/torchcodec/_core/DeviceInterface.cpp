// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DeviceInterface.h"
#include <map>
#include <mutex>
#include "StableABICompat.h"

extern "C" {
#include <libavutil/pixdesc.h>
}

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
// TODO_STABLE_ABI: we might need to support more device types, i.e. those from
// https://github.com/pytorch/pytorch/blob/main/torch/headeronly/core/DeviceType.h
// Ideally we'd remove this helper?
StableDeviceType parseDeviceType(const std::string& deviceType) {
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
    const std::string& device,
    const std::string& variant) {
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
    const std::string_view variant) {
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
  auto format = static_cast<AVPixelFormat>(avFrame->format);
  STD_TORCH_CHECK(
      format == AV_PIX_FMT_RGB24 || format == AV_PIX_FMT_RGB48,
      "Expected RGB24 or RGB48 format, got ",
      (av_get_pix_fmt_name(format) ? av_get_pix_fmt_name(format) : "unknown"));

  int height = avFrame->height;
  int width = avFrame->width;
  AVFrame* avFrameClone = av_frame_clone(avFrame.get());

  auto deleter = [avFrameClone](void*) {
    UniqueAVFrame avFrameToDelete(avFrameClone);
  };

  std::vector<int64_t> shape = {height, width, 3};

  // RGB48 stores 2 bytes per channel (uint16); RGB24 stores 1 byte (uint8).
  // linesize is in bytes, but torch strides are in elements, so divide.
  int bytesPerElement = (format == AV_PIX_FMT_RGB48) ? 2 : 1;
  auto dtype = (format == AV_PIX_FMT_RGB48) ? kStableUInt16 : kStableUInt8;
  std::vector<int64_t> strides = {
      avFrameClone->linesize[0] / bytesPerElement, 3, 1};

  return torch::stable::from_blob(
      avFrameClone->data[0],
      shape,
      strides,
      StableDevice(kStableCPU),
      dtype,
      deleter);
}

} // namespace facebook::torchcodec
