#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs the rocJPEG SDK (GPU JPEG decoder), the ROCm counterpart of nvJPEG.
#
# rocJPEG is NOT preinstalled in the pytorch/manylinux2_28-builder:rocmX.Y
# images (torchvision's build logs show it silently building "without ROCJPEG
# support" there), even though ROCm itself is. Both the build (to compile
# DecodeJpegRocm.cpp) and the runtime (we don't bundle librocjpeg into the wheel)
# need it, so we install it from the ROCm dnf repo that ships in those images.
# libva-amdgpu-devel is a rocJPEG dependency. Mirrors pytorch/vision's
# "Install rocJPEG SDK" step.

set -euo pipefail

# Diagnostics: which repos are configured and whether the rocJPEG/libva packages
# are visible. If this script fails again, these tell us whether it's a
# missing/stale repo vs. a genuinely absent package. Kept non-fatal.
echo "::group::rocJPEG: dnf repos and package availability"
dnf repolist --all 2>&1 | head -40 || true
dnf list --refresh --available 'rocjpeg*' 'libva-amdgpu*' 2>&1 | head -30 || true
echo "::endgroup::"

# rocjpeg-devel is in the ROCm dnf repo; its VA-API dependency (libva-amdgpu*)
# is in the separate amdgpu-graphics repo. --refresh forces a metadata download:
# the manylinux build images ship with stale dnf cache that can miss
# libva-amdgpu-devel ("No match for argument"). If the amdgpu repo still isn't
# visible, fall back to installing rocjpeg-devel alone and let dnf resolve its
# deps (surfacing a clear error if libva genuinely can't be satisfied).
dnf install -y --refresh libva-amdgpu-devel rocjpeg-devel \
    || dnf install -y --refresh rocjpeg-devel

# Diagnostics: where rocJPEG landed (feeds the CMake discovery), and the real
# header API surface. DecodeJpegRocm.cpp is written against the documented API;
# if it fails to compile, these grep results show the actual enum/struct/function
# names from the installed rocjpeg.h so we can fix mismatches without guessing.
echo "::group::rocJPEG: installed files and header API"
rpm -ql rocjpeg-devel rocjpeg 2>&1 | grep -Ei 'rocjpeg\.h|librocjpeg' || true
ls -l /opt/rocm*/lib/librocjpeg* 2>/dev/null || true
rocjpeg_header="$(find /opt -name rocjpeg.h 2>/dev/null | head -1)"
echo "rocjpeg.h -> ${rocjpeg_header:-<not found>}"
if [[ -n "${rocjpeg_header}" ]]; then
    grep -nE \
        'ROCJPEG_OUTPUT_|ROCJPEG_CSS_|ROCJPEG_BACKEND_|ROCJPEG_MAX_COMPONENT|RocJpegDecodeParams|crop_rectangle|output_format|rocJpegDecodeBatched|rocJpegDecode\(|rocJpegGetImageInfo|rocJpegStreamParse|rocJpegCreate|rocJpegGetErrorName' \
        "${rocjpeg_header}" || true
fi
echo "::endgroup::"
