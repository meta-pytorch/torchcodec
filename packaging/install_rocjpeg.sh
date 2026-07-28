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

dnf install -y libva-amdgpu-devel rocjpeg-devel
