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

# Diagnostics: which repos are configured and whether the rocJPEG/libva packages
# are visible. If this script fails again, these tell us whether it's a
# missing/stale repo vs. a genuinely absent package. Kept non-fatal.
echo "::group::rocJPEG: dnf repos and package availability"
dnf repolist --all 2>&1 | head -40 || true
dnf list --refresh --available 'rocjpeg*' 'libva-amdgpu*' 2>&1 | head -30 || true
echo "::endgroup::"

# rocjpeg-devel (in the ROCm repo) requires "(libva-devel >= 2.16.0 or
# libva-amdgpu-devel)". libva-amdgpu-devel lives in the amdgpu-graphics repo,
# which is NOT configured in the manylinux build image (only 'amdgpu' + 'ROCm'
# are), and AlmaLinux 8's libva-devel is older than 2.16.0 -- so a plain
# `dnf install rocjpeg-devel` can't satisfy that dep here.
#
# Preferred path: if libva-amdgpu-devel is reachable, do a normal install.
# Fallback (build image): install rocjpeg's files with rpm --nodeps. We only
# need rocjpeg.h + librocjpeg.so to compile/link; librocjpeg's runtime VA-API
# dependency resolves by the libva.so.2 SONAME, which base AlmaLinux 'libva'
# provides (the >= 2.16.0 rpm constraint is stricter than the SONAME we link
# against). Full hardware VA-API decode at runtime still needs the ROCm graphics
# stack present, but that's a runtime concern, not a build one.
install_rocjpeg_nodeps() {
    echo "libva-amdgpu-devel unavailable in this image; installing rocJPEG via rpm --nodeps."
    # base libva provides libva.so.2 for linking librocjpeg.
    dnf install -y --refresh libva || true
    dnf install -y "dnf-command(download)" >/dev/null 2>&1 \
        || dnf install -y dnf-plugins-core
    local rpm_dir
    rpm_dir="$(mktemp -d)"
    dnf download --destdir "${rpm_dir}" rocjpeg rocjpeg-devel
    rpm -Uvh --nodeps "${rpm_dir}"/rocjpeg*.rpm
}

dnf install -y --refresh libva-amdgpu-devel rocjpeg-devel \
    || install_rocjpeg_nodeps

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
