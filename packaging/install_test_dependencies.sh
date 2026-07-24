#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script installs the test dependencies needed to run the test suite.
#
# Example usage:
#   install_test_dependencies.sh

set -euo pipefail

echo "Installing test dependencies..."
# Ideally we would find a way to get those dependencies from pyproject.toml
python -m pip install numpy pytest pillow

# NOTE: we deliberately do NOT install libheif here. libheif is an optional,
# user-supplied runtime dependency (it's LGPL and never bundled), and
# `conda install libheif` is unsafe in the shared test envs: it pulls libavif16
# (-> aom/svt-av1) which conflicts with the pinned FFmpeg 4 / cuda-toolkit in the
# CUDA and older-FFmpeg matrix cells, and some test shells (e.g. Windows CPU
# smoke) don't even have conda on PATH. Instead, the HEIC tests are marked
# `needs_heic` and simply SKIP when libheif is absent. A dedicated, FFmpeg-free
# CI job (see install-and-test-heic) installs libheif in a clean env and runs
# them with FAIL_WITHOUT_HEIC=1. See packaging/install_libheif.sh.

echo "Test dependencies installed successfully!"
