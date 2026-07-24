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
# FFMPEG_VERSION (optional) controls whether we install libheif, the optional
# HEIC runtime dependency (LGPL, never bundled): we install it into the SAME env
# so HEIC is tested like a real user would have it, EXCEPT for FFmpeg 4 (whose
# old aom/svt-av1 pins are unsatisfiable with conda's libheif) and when no
# version is passed (e.g. the FFmpeg-free jobs), where the HEIC tests just skip.

set -euo pipefail

ffmpeg_version="${1:-}"

echo "Installing test dependencies..."
# Ideally we would find a way to get those dependencies from pyproject.toml
python -m pip install numpy pytest pillow

if [[ -n "${ffmpeg_version}" && "${ffmpeg_version}" != "4" ]]; then
    # Install libheif into the same env, unless it's already there (the CUDA jobs
    # install it at env-creation, because a post-hoc `conda install` re-solves
    # the env and trips the pinned cuda-toolkit; we must not reinstall there).
    # The install is guarded on conda being callable, but FAIL_WITHOUT_HEIC below
    # is set unconditionally: HEIC is meant to be tested here, so if libheif ends
    # up missing the tests fail loudly rather than skipping silently.
    if command -v conda >/dev/null 2>&1; then
        if ! conda list 2>/dev/null | grep -qiE "^libheif[[:space:]]"; then
            conda install -y libheif -c conda-forge
        fi
    fi

    # Make the conda libheif discoverable to the loader in later steps, and
    # require the HEIC tests (FAIL_WITHOUT_HEIC) rather than let them silently
    # skip. On Linux/macOS we prepend nothing to torch's libs but add the conda
    # lib dir to the loader search path (ahead of any system libheif). On
    # Windows, load_heic_library() finds libheif.dll via
    # os.add_dll_directory($CONDA_PREFIX/Library/bin), so no path plumbing needed.
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        case "$(uname -s)" in
            Linux*)  echo "LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" >> "${GITHUB_ENV}" ;;
            Darwin*) echo "DYLD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${DYLD_LIBRARY_PATH:-}" >> "${GITHUB_ENV}" ;;
        esac
        echo "FAIL_WITHOUT_HEIC=1" >> "${GITHUB_ENV}"
    fi
fi

echo "Test dependencies installed successfully!"
