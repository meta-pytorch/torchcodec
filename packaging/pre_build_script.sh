#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# We need to install pybind11 because we need its CMake helpers in order to
# compile correctly on Mac. Pybind11 is actually a C++ header-only library,
# and PyTorch actually has it included. PyTorch, however, does not have the
# CMake helpers.
conda install -y pybind11 -c conda-forge
