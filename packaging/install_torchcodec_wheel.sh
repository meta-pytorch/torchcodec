#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script finds and installs a torchcodec wheel from the dist directory. The
# wheel is expected to have been built and downloaded from a separate job.
#
# Usage:
#   install_torchcodec_wheel.sh [WHEEL_PATTERN]
#
# Example usage:
#   install_torchcodec_wheel.sh
#   install_torchcodec_wheel.sh "*.whl"
#   install_torchcodec_wheel.sh "*cu126-cp310*.whl"

set -euo pipefail

WHEEL_PATTERN="${1:-*.whl}"

wheel_path=$(find pytorch/torchcodec/dist -type f -name "$WHEEL_PATTERN")

if [ -z "$wheel_path" ]; then
  echo "Error: No wheel found matching pattern '$WHEEL_PATTERN' in pytorch/torchcodec/dist/"
  exit 1
fi

echo "Installing $wheel_path"
python -m pip install "$wheel_path" -vvv
