#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Install the build-backend dependencies (scikit-build-core, ninja, pybind11).
# We build the wheel against the pre-built FFmpeg from S3
# (BUILD_AGAINST_ALL_FFMPEG_FROM_S3), so pkg-config is not needed here.
bash packaging/install_build_dependencies.sh
