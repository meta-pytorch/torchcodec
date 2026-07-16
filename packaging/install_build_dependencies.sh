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

set -ex

conda install -y pybind11 -c conda-forge
python -m pip install "scikit-build-core>=0.10" ninja
