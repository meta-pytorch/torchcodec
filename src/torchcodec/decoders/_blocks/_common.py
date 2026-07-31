# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# On CUDA, the blocks only support the NVDEC backend, whose internal variant
# name is "default". The FFmpeg-hwaccel CUDA backend produces frames the blocks
# can't hand out, so set_cuda_backend() intentionally has no effect here.
_DEVICE_VARIANT = "default"
