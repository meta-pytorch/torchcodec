#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script sets up the test environment after installing the torchcodec wheel.
# It installs test dependencies and deletes the src/ folder to ensure tests run
# against the installed wheel rather than local source code.
#
# Example usage:
#   setup_test_env.sh

set -euo pipefail

echo "Installing test dependencies..."
# Ideally we would find a way to get those dependencies from pyproject.toml
python -m pip install numpy pytest pillow

echo "Deleting src/ folder to ensure tests use installed wheel..."
# The only reason we checked-out the repo is to get access to the
# tests. We don't care about the rest. Out of precaution, we delete
# the src/ folder to be extra sure that we're running the code from
# the installed wheel rather than from the source.
# This is just to be extra cautious and very overkill because a)
# there's no way the `torchcodec` package from src/ can be found from
# the PythonPath: the main point of `src/` is precisely to protect
# against that and b) if we ever were to execute code from
# `src/torchcodec`, it would fail loudly because the built .so files
# aren't present there.
rm -r src/
ls

echo "Test environment setup complete!"
