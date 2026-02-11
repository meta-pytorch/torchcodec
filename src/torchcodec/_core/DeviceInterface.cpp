// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DeviceInterface.h"
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>

namespace facebook::torchcodec {

namespace {
using DeviceInterfaceMap =
    std::map<DeviceInterfaceKey, CreateDeviceInterfaceFn>;
static std::mutex g_interface_mutex;

DeviceInterfaceMap& getDeviceMap() {
  static DeviceInterfaceMap deviceMap;
  return deviceMap;
}

// Map from data pointer to AVFrame pointer for cleanup.
// This is needed because the stable ABI's from_blob deleter receives the data
// pointer, but we need to free the AVFrame (which owns the data).
static std::mutex g_avframe_mutex;
static std::unordered_map<void*, AVFrame*>& getAVFrameMap() {
  static std::unordered_map<void*, AVFrame*> avframeMap;
  return avframeMap;
}

void avFrameDeleter(void* data) {
  std::scoped_lock lock(g_avframe_mutex);
  auto& avframeMap = getAVFrameMap();
  auto it = avframeMap.find(data);
  if (it != avframeMap.end()) {
    av_frame_free(&it->second);
    avframeMap.erase(it);
  }
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

StableTensor rgbAVFrameToTensor(const UniqueAVFrame& avFrame) {
  STD_TORCH_CHECK(
      avFrame->format == AV_PIX_FMT_RGB24,
      "Expected RGB24 format, got: ",
      avFrame->format);

  int height = avFrame->height;
  int width = avFrame->width;
  std::vector<int64_t> shape = {height, width, 3};
  std::vector<int64_t> strides = {avFrame->linesize[0], 3, 1};

  // Clone the AVFrame so we own the data
  AVFrame* avFrameClone = av_frame_clone(avFrame.get());
  STD_TORCH_CHECK(avFrameClone != nullptr, "Failed to clone AVFrame");

  // Register the AVFrame for cleanup when the tensor is destroyed.
  // The stable ABI's from_blob deleter receives the data pointer, but we need
  // to free the AVFrame (which owns the data). We use a map to track this.
  {
    std::scoped_lock lock(g_avframe_mutex);
    getAVFrameMap()[avFrameClone->data[0]] = avFrameClone;
  }

  return torch::stable::from_blob(
      avFrameClone->data[0],
      StableIntArrayRef(shape.data(), shape.size()),
      StableIntArrayRef(strides.data(), strides.size()),
      StableDevice(kStableCPU),
      kStableUInt8,
      avFrameDeleter);
}

} // namespace facebook::torchcodec
