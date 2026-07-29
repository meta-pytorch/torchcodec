// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "IOInterface.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

FORCE_PUBLIC_VISIBILITY void
encode_jpeg(
    const torch::stable::Tensor& img,
    int64_t quality,
    IOInterface& interface);

} // namespace facebook::torchcodec
