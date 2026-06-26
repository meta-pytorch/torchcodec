// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>

#include "SingleStreamDecoder.h"

// Torch-free serialization of decoder metadata to JSON strings. These were
// previously inline in custom_ops.cpp (the torch adapter); they live here so the
// torch-free pybind frontend can read metadata too. They operate on a plain
// SingleStreamDecoder* and depend only on the (torch-free) metadata structs.

namespace facebook::torchcodec {

// Best-stream-oriented summary metadata (the "get_json_metadata" op).
std::string getVideoJsonMetadata(SingleStreamDecoder* decoder);

// Container-level metadata.
std::string getContainerJsonMetadata(SingleStreamDecoder* decoder);

// Per-stream metadata.
std::string getStreamJsonMetadata(
    SingleStreamDecoder* decoder,
    int64_t streamIndex);

} // namespace facebook::torchcodec
