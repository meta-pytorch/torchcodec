// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "EncodingDeviceInterface.h"

#include <map>
#include <mutex>

namespace facebook::torchcodec {

namespace {

using EncodingDeviceInterfaceMap =
    std::map<EncodingDeviceInterfaceKey, CreateEncodingDeviceInterfaceFn>;
static std::mutex g_interface_mutex;

EncodingDeviceInterfaceMap& getDeviceMap() {
  static EncodingDeviceInterfaceMap deviceMap;
  return deviceMap;
}

} // namespace

bool registerEncodingDeviceInterface(
    const EncodingDeviceInterfaceKey& key,
    CreateEncodingDeviceInterfaceFn createInterface) {
  std::scoped_lock lock(g_interface_mutex);
  EncodingDeviceInterfaceMap& deviceMap = getDeviceMap();

  TORCH_CHECK(
      deviceMap.find(key) == deviceMap.end(),
      "Encoding device interface already registered for device type ",
      key.deviceType);
  deviceMap.insert({key, createInterface});

  return true;
}

std::unique_ptr<EncodingDeviceInterface> createEncodingDeviceInterface(
    const torch::Device& device) {
  EncodingDeviceInterfaceKey key(device.type());
  std::scoped_lock lock(g_interface_mutex);
  EncodingDeviceInterfaceMap& deviceMap = getDeviceMap();

  auto it = deviceMap.find(key);
  if (it != deviceMap.end()) {
    return std::unique_ptr<EncodingDeviceInterface>(it->second(device));
  }

  TORCH_CHECK(
      false,
      "No encoding device interface found for device type: ",
      device.type());
}

} // namespace facebook::torchcodec
