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

# A plain `dnf install rocjpeg-devel` fails in the manylinux CI image:
# rocjpeg-devel needs libva >= 2.16, which isn't available there. So install its
# files with --nodeps -- we only need rocjpeg.h + librocjpeg.so, and base libva
# provides the libva.so.2 soname we link against.
dnf install -y --refresh libva
dnf install -y "dnf-command(download)" >/dev/null 2>&1 || dnf install -y dnf-plugins-core
rpm_dir="$(mktemp -d)"
dnf download --destdir "${rpm_dir}" rocjpeg rocjpeg-devel
rpm -Uvh --nodeps "${rpm_dir}"/rocjpeg*.rpm
