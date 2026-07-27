#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs the test dependencies needed to run the test suite.
#
# Usage:
#   install_test_dependencies.sh [FFMPEG_VERSION]
#
# FFMPEG_VERSION (optional) selects whether we install libheif, the optional
# HEIC runtime dependency: we install it for every version except FFmpeg 4,
# whose old aom/svt-av1 pins can't be satisfied alongside conda's libheif (so we
# don't test HEIC on FFmpeg 4).

set -euo pipefail

ffmpeg_version="${1:-}"

echo "Installing test dependencies..."
# Ideally we would find a way to get those dependencies from pyproject.toml
python -m pip install numpy pytest pillow

if [[ -n "${ffmpeg_version}" && "${ffmpeg_version}" != "4" ]]; then
    # --freeze-installed keeps the already-installed packages untouched and only
    # adds libheif + its deps. Without it, conda re-solves the whole env, which
    # leads to failures on the CUDA jobs.
    conda install -y --freeze-installed libheif -c conda-forge
fi

echo "Test dependencies installed successfully!"
