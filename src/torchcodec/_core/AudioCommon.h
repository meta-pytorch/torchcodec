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

// Decoded audio always leaves torchcodec as normalized float32 samples,
// one plane per channel.
constexpr AVSampleFormat kAudioOutSampleFormat = AV_SAMPLE_FMT_FLTP;

// The dtype that holds `sample_format`'s samples exactly. Planar and packed
// variants of a format share a sample type, which is why this doesn't care
// which one it is given.
torch::headeronly::ScalarType sample_format_dtype(AVSampleFormat sample_format);

// The inverse: the sample format that a contiguous [num_channels, num_samples]
// tensor of `dtype` already is. One row per channel is precisely what planar
// means, which is what lets swresample read such a tensor's rows directly.
AVSampleFormat planar_sample_format(torch::headeronly::ScalarType dtype);

// Runs swr_convert() straight into a fresh float32 [num_channels, N] tensor,
// narrowed to the number of samples it actually produced - which is at most
// `num_out_samples_bound` and is often fewer, since swresample holds back the
// samples that need future input. Pass a null `src_planes` to flush those out.
torch::stable::Tensor swr_convert_to_tensor(
    const UniqueSwrContext& swr_context,
    const uint8_t* const* src_planes,
    int num_src_samples,
    int num_out_channels,
    int64_t num_out_samples_bound);

} // namespace facebook::torchcodec
