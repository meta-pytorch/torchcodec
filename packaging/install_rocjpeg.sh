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

# rocJPEG decodes via VA-API, so at runtime it needs the AMD VA-API driver
# (mesa-amdgpu-va-drivers + libva-amdgpu), not just librocjpeg -- without it,
# vaInitialize() fails and decoding errors out. A proper `dnf install` pulls that
# whole stack, but only where AMD's amdgpu-graphics repo is configured (the ROCm
# test runners). On the plain build image that repo is absent; there we only need
# to *compile* against rocjpeg.h/librocjpeg.so, so fall back to installing just
# those with --nodeps (base libva provides the libva.so.2 soname we link).
#
# ROCm >= 7.14 distributes the full ROCm stack (including rocJPEG) as pip wheels
# (_rocm_sdk_core / _rocm_sdk_devel site-packages). In that case librocjpeg.so
# and rocjpeg.h are already present and the dnf packages don't exist, so we
# skip the install entirely.
install_rocjpeg_build_only() {
    dnf install -y --refresh libva
    dnf install -y "dnf-command(download)" >/dev/null 2>&1 || dnf install -y dnf-plugins-core
    rpm_dir="$(mktemp -d)"
    dnf download --destdir "${rpm_dir}" rocjpeg rocjpeg-devel
    rpm -Uvh --nodeps "${rpm_dir}"/rocjpeg*.rpm
}

# Check if librocjpeg is already available (e.g. via ROCm 7.14+ pip wheels).
if python3 -c "
import glob, sys
# _rocm_sdk_core and _rocm_sdk_devel are the pip-wheel-based ROCm installs
hits = (glob.glob('/opt/conda/**/librocjpeg.so*', recursive=True) +
        glob.glob('/opt/rocm/lib/librocjpeg.so*'))
sys.exit(0 if hits else 1)
" 2>/dev/null; then
    echo "librocjpeg already present (ROCm pip-wheel install); skipping dnf install."
else
    dnf install -y --refresh rocjpeg-devel libva-amdgpu mesa-amdgpu-va-drivers \
        || install_rocjpeg_build_only
fi
