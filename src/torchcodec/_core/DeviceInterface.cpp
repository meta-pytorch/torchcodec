// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DeviceInterface.h"
#include <ATen/ops/from_blob.h>
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
      av_get_pix_fmt_name(format));

  int height = avFrame->height;
  int width = avFrame->width;
  AVFrame* avFrameClone = av_frame_clone(avFrame.get());

  // TODO_STABLE_ABI: we're still using the non-stable ABI here. That's because
  // stable::from_blob doesn't yet support a capturing lambda deleter. We need
  // to land https://github.com/pytorch/pytorch/pull/175089.
  // TC won't be able stable until this is resolved.
  auto deleter = [avFrameClone](void*) {
    UniqueAVFrame avFrameToDelete(avFrameClone);
  };

  at::Tensor tensor;
  if (format == AV_PIX_FMT_RGB48) {
    // RGB48: 6 bytes per pixel (3 channels x 2 bytes each).
    // Strides are in uint16 elements: linesize is in bytes, so divide by 2.
    std::vector<int64_t> shape = {height, width, 3};
    std::vector<int64_t> strides = {avFrameClone->linesize[0] / 2, 3, 1};
    tensor = at::from_blob(
        avFrameClone->data[0],
        shape,
        strides,
        deleter,
        {at::ScalarType::UInt16});
  } else {
    // RGB24: 3 bytes per pixel.
    std::vector<int64_t> shape = {height, width, 3};
    std::vector<int64_t> strides = {avFrameClone->linesize[0], 3, 1};
    tensor = at::from_blob(
        avFrameClone->data[0], shape, strides, deleter, {at::kByte});
  }

  // We got an at::Tensor, we have to convert it to a torch::stable::Tensor.
  // This is safe, there won't be any memory leak, i.e. the at::Tensor's deleter
  // will properly be passed down to the torch::stable::Tensor.
  at::Tensor* p = new at::Tensor(std::move(tensor));
  return torch::stable::Tensor(reinterpret_cast<AtenTensorHandle>(p));
}

} // namespace facebook::torchcodec
