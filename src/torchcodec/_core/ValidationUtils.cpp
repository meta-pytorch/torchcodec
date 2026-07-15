// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ValidationUtils.h"
#include <limits>
#include "StableABICompat.h"

namespace facebook::torchcodec {

int validate_int64_to_int(int64_t value, const std::string& parameter_name) {
  STD_TORCH_CHECK(
      value >= std::numeric_limits<int>::min() &&
          value <= std::numeric_limits<int>::max(),
      parameter_name,
      "=",
      value,
      " is out of range for int type.");

  return static_cast<int>(value);
}

std::optional<int> validate_optional_int64_to_int(
    const std::optional<int64_t>& value,
    const std::string& parameter_name) {
  if (value.has_value()) {
    return validate_int64_to_int(value.value(), parameter_name);
  } else {
    return std::nullopt;
  }
}

std::streampos validate_uint64_to_streampos(
    uint64_t value,
    const std::string& parameter_name) {
  // We validate against streamoff limits because streampos
  // (std::fpos<state_type>) stores the actual position as streamoff internally.
  // https://en.cppreference.com/w/cpp/io/fpos.html
  STD_TORCH_CHECK(
      value <=
          static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()),
      parameter_name,
      "=",
      value,
      " is out of range for streampos type.");

  return static_cast<std::streampos>(value);
}

} // namespace facebook::torchcodec
