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
# ROCm >= 7.14 distributes the full ROCm stack (including rocJPEG and mesa)
# as pip wheels (_rocm_sdk_core / _rocm_sdk_devel site-packages). In that case
# librocjpeg.so and rocjpeg.h are already present and the AMD VA-API backend
# driver (mesa) is bundled inside _rocm_sdk_core — no separate dnf install
# needed. We only need the base libva soname for the dynamic linker.
install_rocjpeg_build_only() {
    dnf install -y --refresh libva
    dnf install -y "dnf-command(download)" >/dev/null 2>&1 || dnf install -y dnf-plugins-core
    rpm_dir="$(mktemp -d)"
    dnf download --destdir "${rpm_dir}" rocjpeg rocjpeg-devel
    rpm -Uvh --nodeps "${rpm_dir}"/rocjpeg*.rpm
}

# Check if librocjpeg is already available via the ROCm >= 7.14 pip-wheel layout
# (_rocm_sdk_core / _rocm_sdk_devel under site-packages).  In that layout AMD
# bundles mesa (the VA-API DRI driver) inside _rocm_sdk_core, so no separate
# dnf install of mesa-amdgpu-va-drivers is needed — only the base libva soname.
#
# NOTE: we intentionally do NOT match /opt/rocm/lib/librocjpeg.so* here.  If
# librocjpeg was installed via the ROCm <= 7.2 system RPM path it may be
# present at /opt/rocm but mesa-amdgpu-va-drivers may not be — they are
# separate packages and must be installed together.  Let the else branch handle
# that case so it always installs the full VA-API stack.
if python3 -c "
import glob, sys
# Only the pip-wheel layout bundles mesa alongside librocjpeg.
hits = glob.glob('/opt/conda/**/librocjpeg.so*', recursive=True)
sys.exit(0 if hits else 1)
" 2>/dev/null; then
    echo "librocjpeg already present via ROCm pip-wheel install; skipping dnf install."
    dnf install -y libva 2>/dev/null || true
else
    # Covers both first-time installs and the ROCm <= 7.2 system-RPM path
    # (dnf is idempotent for already-installed packages).
    dnf install -y --refresh rocjpeg-devel libva-amdgpu mesa-amdgpu-va-drivers \
        || install_rocjpeg_build_only
fi
