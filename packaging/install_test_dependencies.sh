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

echo "Test dependencies installed successfully!"
