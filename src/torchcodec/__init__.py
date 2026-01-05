# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import os
from pathlib import Path

# Note: usort wants to put Frame and FrameBatch after decoders and samplers,
# but that results in circular import.
from ._frame import AudioSamples, Frame, FrameBatch  # usort:skip # noqa
from . import decoders, encoders, samplers, transforms  # noqa

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


def _import_device_backends():
    """
    Leverage the Python plugin mechanism to load out-of-the-tree device extensions.
    """
    from importlib.metadata import entry_points

    group_name = "torchcodec.backends"
    backend_extensions = entry_points(group=group_name)

    for backend_extension in backend_extensions:
        try:
            # Load the extension
            entrypoint = backend_extension.load()
            # Call the entrypoint
            entrypoint()
        except Exception as err:
            raise RuntimeError(
                f"Failed to load the backend extension: {backend_extension.name}. "
                f"You can disable extension auto-loading with TORCHCODEC_DEVICE_BACKEND_AUTOLOAD=0."
            ) from err


def _is_device_backend_autoload_enabled() -> builtins.bool:
    """
    Whether autoloading out-of-the-tree device extensions is enabled.
    The switch depends on the value of the environment variable
    `TORCHCODEC_DEVICE_BACKEND_AUTOLOAD`.

    Returns:
        bool: Whether to enable autoloading the extensions. Enabled by default.
    """
    # enabled by default
    return os.getenv("TORCHCODEC_DEVICE_BACKEND_AUTOLOAD", "1") == "1"


# `_import_device_backends` should be kept at the end to ensure
# all the other functions in this module that may be accessed by
# an autoloaded backend are defined
if _is_device_backend_autoload_enabled():
    _import_device_backends()
