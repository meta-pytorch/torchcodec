# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

# Note: usort wants to put Frame and FrameBatch after decoders and samplers,
# but that results in circular import.
from ._frame import AudioSamples, Frame, FrameBatch  # usort:skip # noqa
from . import decoders  # noqa

try:
    import torch as _torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    # Encoders, samplers and transforms operate on torch tensors and require
    # torch. On a torch-free install only video decoding is available.
    from . import encoders, samplers, transforms  # noqa

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
