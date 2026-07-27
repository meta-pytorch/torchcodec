#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs the test dependencies needed to run the test suite.
#
# Usage:
#   install_test_dependencies.sh [INSTALL_LIBHEIF]
#
# INSTALL_LIBHEIF: whether to install libheif, the optional HEIC runtime
# dependency (LGPL, never bundled). Defaults to "true"; callers pass "false"
# where it can't be installed: on FFmpeg 4 (its aom/svt-av1 pins can't be
# satisfied alongside conda's libheif) and on some CUDA jobs whose conda env can
# no longer be re-solved.

set -euo pipefail

install_libheif="${1:-true}"

echo "Installing test dependencies..."
python -m pip install numpy pytest pillow

if [[ "${install_libheif}" == "true" ]]; then
    conda install -y libheif -c conda-forge
fi

echo "Test dependencies installed successfully!"
