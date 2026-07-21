#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs the build-time dependencies needed to *compile* torchcodec: the
# scikit-build-core build backend, the Ninja generator it drives, and pybind11.
# Because we build with --no-build-isolation (torch must come from a custom index,
# so it can't live in pyproject's build-system.requires), pip does not install
# build-system.requires for us -- we must do it here.
#
# This script intentionally does NOT install:
# - torch: install it separately (a nightly matching your CPU/CUDA variant).
# - FFmpeg: a runtime/build dependency handled by the caller.
# - pkg-config: only needed to *locate an installed FFmpeg* at build time, i.e.
#   builds that don't set BUILD_AGAINST_ALL_FFMPEG_FROM_S3. Callers that build
#   against an installed FFmpeg install pkg-config alongside it.
#
# The image decoder libs (libjpeg-turbo, libpng, libwebp) ARE installed here by
# default so the image decoders get built into the wheels. Builds that don't need
# image decoding (e.g. the mypy CI job) can skip them by setting
# TORCHCODEC_SKIP_IMAGE_DEPS=1, in which case the decoders compile as no-op stubs.

set -ex

conda install -y pybind11 -c conda-forge
python -m pip install "scikit-build-core>=0.10" ninja

if [[ "${TORCHCODEC_SKIP_IMAGE_DEPS:-0}" != "1" ]]; then
    conda install -y libjpeg-turbo -c pytorch
    conda install -y libpng -c conda-forge
    # Pin >=1.3: CMake has no built-in FindWebP, so we rely on config mode
    # (find_package(WebP)), and only libwebp >=1.3 conda-forge builds ship
    # WebPConfig.cmake. Older builds (e.g. 1.2.4) would silently disable WebP.
    conda install -y "libwebp>=1.3" -c conda-forge
fi
