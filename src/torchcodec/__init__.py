# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

# Note: The seemingly redundant `x as x` aliases mark these as public re-exports.
# Without them, type-checkers consider them private, and report errors when users
# access e.g. `torchcodec.Frame` (mypy with `--strict`, pyright by default).
# https://typing.python.org/en/latest/spec/distributing.html#import-conventions

# Note: usort wants to put Frame and FrameBatch after decoders and samplers,
# but that results in circular import.
from ._frame import (
    AudioSamples as AudioSamples,
    Frame as Frame,
    FrameBatch as FrameBatch,
)  # usort:skip
from . import (
    decoders as decoders,
    encoders as encoders,
    samplers as samplers,
    transforms as transforms,
)

try:
    # Note that version.py is generated during install.
    from .version import __version__ as __version__
except Exception:
    pass

# cmake_prefix_path is needed for downstream cmake-based builds that use
# torchcodec as a dependency to tell cmake where torchcodec is installed and where to find its
# CMake configuration files.
# Pytorch itself has a similar mechanism which we use in our setup.py!
cmake_prefix_path = Path(__file__).parent / "share" / "cmake"
# Similarly, these are exposed for downstream builds that use torchcodec as a
# dependency.
from ._core import (
    core_library_path as core_library_path,
    ffmpeg_major_version as ffmpeg_major_version,
)  # usort:skip


# Leverage the Python plugin mechanism to load out-of-the-tree device extensions.
# See https://github.com/pytorch/pytorch/pull/127074 that enabled the same
# plugin support in torch.
# This block should be kept at the end to ensure all the other functions in this
# module that may be accessed by an autoloaded backend are defined.
_TORCHCODEC_DEVICE_BACKEND_AUTOLOAD_VAR_NAME = "TORCHCODEC_DEVICE_BACKEND_AUTOLOAD"
if os.getenv(_TORCHCODEC_DEVICE_BACKEND_AUTOLOAD_VAR_NAME, "1") == "1":
    from importlib.metadata import entry_points

    _backend_extensions = entry_points(group="torchcodec.backends")

    for _backend_extension in _backend_extensions:
        try:
            _entrypoint = _backend_extension.load()
            _entrypoint()
        except Exception as _err:
            raise RuntimeError(
                f"Failed to load the backend extension: {_backend_extension.name}. "
                f"You can disable extension auto-loading with {_TORCHCODEC_DEVICE_BACKEND_AUTOLOAD_VAR_NAME}=0."
            ) from _err
