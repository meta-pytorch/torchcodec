// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ValidationUtils.h"
#include <limits>
#include "StableABICompat.h"

namespace facebook::torchcodec {

int validateInt64ToInt(int64_t value, const std::string& parameterName) {
  STD_TORCH_CHECK(
      value >= std::numeric_limits<int>::min() &&
          value <= std::numeric_limits<int>::max(),
      parameterName,
      "=",
      value,
      " is out of range for int type.");

  return static_cast<int>(value);
}

std::optional<int> validateOptionalInt64ToInt(
    const std::optional<int64_t>& value,
    const std::string& parameterName) {
  if (value.has_value()) {
    return validateInt64ToInt(value.value(), parameterName);
  } else {
    return std::nullopt;
  }
}

std::streampos validateUint64ToStreampos(
    uint64_t value,
    const std::string& parameterName) {
  // std::streampos is typically std::streamoff, which is signed, so converting
  // it to uuint64_t is a widening cast.
  // https://en.cppreference.com/w/cpp/io/streamoff.html
  STD_TORCH_CHECK(
      value <=
          static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()),
      parameterName,
      "=",
      value,
      " is out of range for streampos type.");

  return static_cast<std::streampos>(value);
}

} // namespace facebook::torchcodec
