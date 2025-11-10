# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

# Note: usort wants to put Frame and FrameBatch after decoders and samplers,
# but that results in circular import.
from ._frame import AudioSamples, Frame, FrameBatch  # usort:skip # noqa
from . import decoders, encoders, samplers  # noqa

try:
    # Note that version.py is generated during install.
    from .version import __version__  # noqa: F401
except Exception:
    pass

# cmake_prefix_path is needed for downstream cmake-based builds that use
# torchcodec as a dependency to tell cmake where torchcodec is installed and where to find its
# CMake configuration files.
# Pytorch itself has a similar mechanism which we use in our setup.py!
cmake_prefix_path = Path(__file__).parent / "share" / "cmake"
# Similarly, these are exposed for downstream builds that use torchcodec as a
# dependency.
from ._core import core_library_path, ffmpeg_major_version  # usort:skip
