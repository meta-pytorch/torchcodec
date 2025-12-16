#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script removes the src/ directory to ensure tests run against the
# installed wheel rather than local source code.
#
# Usage:
#   remove_src.sh

set -euo pipefail

echo "Deleting src/ folder to ensure tests use installed wheel..."
# The only reason we checked-out the repo is to get access to the
# tests and to the helper scripts for the CI. We don't care about the rest.
# Out of precaution, we delete
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

echo "src/ folder removed successfully!"
