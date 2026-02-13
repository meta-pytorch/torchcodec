// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// NVDECCache is a template class with implementation in the header.
// This file is kept for build system compatibility and potential future
// non-template code.

#include "NVDECCache.h"

namespace facebook::torchcodec {

// Explicit template instantiation for the default policy.
// This can reduce compile times when the same instantiation is used
// in multiple translation units.
template class NVDECCacheImpl<DefaultNVDECEvictionPolicy>;

} // namespace facebook::torchcodec
