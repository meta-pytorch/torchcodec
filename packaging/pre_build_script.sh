#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# We build the pybind_ops Python extension module with nanobind (in stable-ABI
# mode), so nanobind must be available at build time: its headers, its bundled
# static library, and its CMake helpers (nanobind_add_module). We install it
# with pip so that `python -m nanobind --cmake_dir` resolves during the CMake
# configure step.
python -m pip install nanobind
