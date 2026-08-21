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

# Skip if rocjpeg is already installed via the ROCm pip-wheel distribution
# (ROCm >= 7.14: librocjpeg ships inside _rocm_sdk_core / _rocm_sdk_devel
# site-packages).
#
# Primary check: importlib.util.find_spec queries Python's import system and
# works in any environment (conda, venv, system Python, etc.).
# Fallback glob: handles the case where _rocm_sdk_* is installed for a
# different Python version than the one running this script (e.g. the script
# runs in a Python 3.10 conda env but _rocm_sdk_devel is under Python 3.11).
# The pattern /*/lib/python*/site-packages covers standard install prefixes
# (/opt/conda, /usr, /usr/local, etc.) without being conda-specific.
if python3 -c "
import importlib.util, pathlib, sys, glob
for pkg in ('_rocm_sdk_core', '_rocm_sdk_devel'):
    spec = importlib.util.find_spec(pkg)
    if spec and spec.submodule_search_locations:
        pkg_root = pathlib.Path(list(spec.submodule_search_locations)[0])
        if (pkg_root / 'include' / 'rocjpeg' / 'rocjpeg.h').exists():
            sys.exit(0)
if glob.glob('/*/lib/python*/site-packages/_rocm_sdk_*/include/rocjpeg/rocjpeg.h'):
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
    echo "rocjpeg already installed via ROCm pip-wheel; skipping dnf install."
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
