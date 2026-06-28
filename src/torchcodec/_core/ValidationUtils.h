// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace facebook::torchcodec {

int validate_int64_to_int(int64_t value, const std::string& parameter_name);

std::optional<int> validate_optional_int64_to_int(
    const std::optional<int64_t>& value,
    const std::string& parameter_name);

std::streampos validate_uint64_to_streampos(
    uint64_t value,
    const std::string& parameter_name);

} // namespace facebook::torchcodec
