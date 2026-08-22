// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "FFMPEGCommon.h"
#include "StableABICompat.h"

// Where FFmpeg's audio samples meet torch tensors. FFMPEGCommon deliberately
// knows nothing about tensors, and these helpers are shared by the decode, the
// conversion and the SingleStreamDecoder paths, so they live on their own.

namespace facebook::torchcodec {

// The dtype that holds `sample_format`'s samples exactly. Planar and packed
// variants of a format share a sample type, which is why this doesn't care
// which one it is given.
torch::headeronly::ScalarType sample_format_dtype(AVSampleFormat sample_format);

} // namespace facebook::torchcodec
