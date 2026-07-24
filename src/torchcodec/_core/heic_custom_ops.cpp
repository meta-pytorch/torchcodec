// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// PyTorch custom-op registration for the HEIC decoder. This lives in the
// libtorchcodec_heic library, which is SEPARATE from libtorchcodec_image: it
// links libheif (LGPL), which we do NOT bundle in our wheels. libheif is an
// optional, user-supplied runtime dependency, so this library is loaded lazily
// (only on the first decode_heic / decode_image-of-HEIC), never at import time.
// See load_heic_library() in _internally_replaced_utils.py.
//
// We register decode_heic into the EXISTING torchcodec_ns namespace via a
// library fragment, so it composes with the ops registered by
// libtorchcodec_image's own fragment.

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
