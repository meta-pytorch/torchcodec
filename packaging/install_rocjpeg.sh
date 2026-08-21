#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs the rocJPEG SDK
#
# rocJPEG is not preinstalled by default in the CI runners. Both the build
# and the test jobs need it (we don't ship it in the wheel, unlike nvjpeg), so
# we install it from the ROCm dnf repo.

set -euo pipefail

# Skip if rocjpeg is already installed (e.g. via the ROCm pip-wheel distribution).
if python3 -c "
import glob, sys
found = (glob.glob('/opt/rocm/include/rocjpeg/rocjpeg.h') or
         glob.glob('/opt/conda/**/rocjpeg.h', recursive=True))
sys.exit(0 if found else 1)
" 2>/dev/null; then
    echo "rocjpeg already installed; skipping dnf install."
    exit 0
fi

# Install from the ROCm dnf repo.
# mesa-amdgpu-va-drivers is declared as an RPM dependency of rocjpeg but may
# not be available as a standalone dnf package (it is installed via
# amdgpu-install as part of the GPU driver stack). Fall back to rpm --nodeps
# if the regular dnf install fails for that reason.
if ! dnf install -y rocjpeg rocjpeg-devel; then
    # Ensure the 'dnf download' subcommand is available.
    dnf install -y "dnf-command(download)" 2>/dev/null || dnf install -y dnf-plugins-core
    tmpdir=$(mktemp -d)
    dnf download --destdir "$tmpdir" rocjpeg rocjpeg-devel
    rpm -Uvh --nodeps "$tmpdir"/*.rpm
    rm -rf "$tmpdir"
fi
