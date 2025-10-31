# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

# Note: usort wants to put Frame and FrameBatch after decoders and samplers,
# but that results in circular import.
from ._core import core_library_path, ffmpeg_major_version
from ._frame import AudioSamples, Frame, FrameBatch  # usort:skip # noqa
from . import decoders, samplers  # noqa

try:
    # Note that version.py is generated during install.
    from .version import __version__  # noqa: F401
except Exception:
    pass

# `torchcodec.cmake_prefix_path` is a Python-based way to programmatically
# obtain the correct CMAKE_PREFIX_PATH value for the TorchCodec installation.
# It can be used in a build system of a C++ application to ensure that CMake
# can successfully find TorchCodec C++ libraries. This variable is exposed
# as TorchCodec API.
cmake_prefix_path = Path(__file__).parent / "share" / "cmake"
