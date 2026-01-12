#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script installs FFmpeg from conda-forge after asserting that FFmpeg is
# not already installed.
#
# Usage:
#   install_ffmpeg.sh FFMPEG_VERSION
#   install_ffmpeg.sh 7.0.1
#   install_ffmpeg.sh 8.0

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Error: Missing required FFmpeg version"
  echo "Usage: install_ffmpeg.sh FFMPEG_VERSION"
  echo "Example: install_ffmpeg.sh 7.0.1"
  exit 1
fi

FFMPEG_VERSION="$1"

# Ideally we would have checked for that before installing the wheel,
# but we need to checkout the repo to access this file, and we don't
# want to checkout the repo before installing the wheel to avoid any
# side-effect. It's OK.
source packaging/helpers.sh
assert_ffmpeg_not_installed

echo "Installing FFmpeg version $FFMPEG_VERSION from conda-forge..."
conda install "ffmpeg=$FFMPEG_VERSION" -c conda-forge
ffmpeg -version
