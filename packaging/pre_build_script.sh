#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

bash packaging/install_build_dependencies.sh

# On ROCm wheel builds (CU_VERSION is e.g. "rocm7.1"), install the rocJPEG SDK so
# the GPU JPEG decoder compiles. It's not preinstalled in the manylinux ROCm
# images. Runs in the same container that then compiles the wheel.
if [[ "${CU_VERSION:-}" == rocm* ]]; then
    bash packaging/install_rocjpeg.sh
fi
