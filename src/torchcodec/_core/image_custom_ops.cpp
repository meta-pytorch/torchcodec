// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// PyTorch custom-op registration for the image decoders. This lives in the
// libtorchcodec_image library, which is FFmpeg-free and loaded in its own
// RTLD_LOCAL symbol group (see load_torchcodec_shared_libraries): that keeps
// our bundled image codec libs (libjpeg/libpng/libwebp) isolated from the codec
// libs pulled in by the user's FFmpeg, so they can't collide.
//
// We register into the SAME torchcodec_ns namespace as the FFmpeg custom ops
// (custom_ops.cpp) using STABLE_TORCH_LIBRARY_FRAGMENT. Both this library and
// custom_ops.cpp use the FRAGMENT variant (rather than STABLE_TORCH_LIBRARY,
// which requires a single exclusive owner of the namespace), so the two
// libraries can be loaded in either order.

#include "DecodeGif.h"
#include "DecodeJpeg.h"
#include "DecodePng.h"
#include "DecodeWebp.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

STABLE_TORCH_LIBRARY_FRAGMENT(torchcodec_ns, m) {
  m.def("decode_jpeg(Tensor data, int mode) -> Tensor");
  m.def("decode_png(Tensor data, int mode) -> Tensor");
  m.def("decode_webp(Tensor data, int mode) -> Tensor");
  m.def("decode_gif(Tensor data, int mode) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(torchcodec_ns, CPU, m) {
  m.impl("decode_jpeg", TORCH_BOX(&decode_jpeg));
  m.impl("decode_png", TORCH_BOX(&decode_png));
  m.impl("decode_webp", TORCH_BOX(&decode_webp));
  m.impl("decode_gif", TORCH_BOX(&decode_gif));
}

} // namespace facebook::torchcodec
