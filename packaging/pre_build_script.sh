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

# We build with `python -m build --no-isolation`, which means the build backend
# (and everything else in build-system.requires) must already be present in the
# current environment - pip/build won't create an isolated env to install them.
# Without this, the build fails with "Backend 'scikit_build_core.build' is not
# available." pybind11 is installed above; `build` is provided by test-infra.
python -m pip install "scikit-build-core>=0.10"
