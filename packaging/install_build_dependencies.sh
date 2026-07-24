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
# Also installs the image decoder libs (libjpeg-turbo, libpng, libwebp, and
# libheif) by default. Builds that don't need image decoding (e.g. the mypy CI
# job) can skip them by setting TORCHCODEC_BUILD_IMAGE=0.
#
# This script intentionally does NOT install:
# - torch: install it separately (a nightly matching your CPU/CUDA variant).
# - FFmpeg: a runtime/build dependency handled by the caller.
# - libavif: unlike the other image libs, it is not installed from conda. The
#   build always fetches a slim decode-only libavif from S3
# - pkg-config: only needed to *locate an installed FFmpeg* at build time, i.e.
#   builds that don't set BUILD_AGAINST_ALL_FFMPEG_FROM_S3. Callers that build
#   against an installed FFmpeg install pkg-config alongside it.

set -ex

conda install -y pybind11 -c conda-forge
python -m pip install "scikit-build-core>=0.10" ninja

if [[ "${TORCHCODEC_BUILD_IMAGE:-ON}" != "0" ]]; then
    conda install -y libjpeg-turbo -c pytorch
    conda install -y libpng -c conda-forge
    conda install -y "libwebp>=1.3" -c conda-forge
    # libheif is *built against* (for libtorchcodec_heic) but, unlike the image
    # libs above, it is NOT bundled into the wheel: it's LGPL and treated as a
    # user-supplied runtime dependency (see packaging/repair_wheel.py, which
    # excludes it). We install it here alongside the other image libs (gated on
    # the same TORCHCODEC_BUILD_IMAGE flag) purely so the build can link the real
    # HEIC decoder; without it the build falls back to a stub. Installing it in
    # the build env is safe (FFmpeg comes from S3, not conda, so there's no
    # aom/svt-av1 pin to conflict with -- unlike the *test* envs).
    conda install -y libheif -c conda-forge

    echo "=== image build deps installed; libheif info: ==="
    conda list 2>/dev/null | grep -iE "libheif|libde265|libavif|libjpeg|libpng|libwebp" || true
fi
