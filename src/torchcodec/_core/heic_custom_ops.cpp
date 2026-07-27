// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecodeHeic.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

STABLE_TORCH_LIBRARY_FRAGMENT(torchcodec_ns, m) {
  m.def("decode_heic(Tensor input, int mode) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(torchcodec_ns, CPU, m) {
  m.impl("decode_heic", TORCH_BOX(&decode_heic));
}

} // namespace facebook::torchcodec
