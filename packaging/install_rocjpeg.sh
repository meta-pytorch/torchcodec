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

# ROCm >= 7.14 distributes the full ROCm stack (including rocJPEG and mesa)
# as pip wheels (_rocm_sdk_core / _rocm_sdk_devel site-packages). In that case
# librocjpeg.so, rocjpeg.h, and the AMD VA-API backend driver (mesa) are all
# bundled inside _rocm_sdk_core — no separate dnf install needed.

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
    # libva is bundled inside _rocm_sdk_core/lib/rocm_sysdeps/lib/ and
    # librocjpeg's own RPATH resolves it from there — no system install needed.
else
    # ROCm <=7.2 system RPM path.
    # rocjpeg-devel's RPM dependencies pull in libva and mesa VA drivers automatically.
    dnf install -y rocjpeg-devel
fi
