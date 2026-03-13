// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "FFMPEGCommon.h"

namespace facebook::torchcodec {

// Returns true if the AVFrame has HDR transfer characteristics (PQ or HLG).
bool isHDRFrame(const AVFrame* frame);

// Converts an HDR AVFrame (PQ or HLG, BT.2020) to an SDR AVFrame in RGB24
// (BT.709). The full pipeline is:
//   1. YUV → RGB using BT.2020 NCL matrix
//   2. PQ EOTF or HLG EOTF (linearization)
//   3. BT.2020 → BT.709 gamut mapping
//   4. Hable tone mapping
//   5. BT.709 OETF + quantize to uint8 RGB24
UniqueAVFrame toneMapHDRFrame(const UniqueAVFrame& src);

} // namespace facebook::torchcodec
