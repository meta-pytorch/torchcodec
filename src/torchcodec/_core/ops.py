# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This module loads the torchcodec shared libraries once, then dispatches the
# core ops to one of two backends:
#   - _torch_ops.py  (when torch is installed): the PyTorch custom ops, with
#     torch.compile support; frames are returned as torch.Tensor. This path is
#     unchanged from the original implementation.
#   - _numpy_ops.py  (torch-free install): the decoder ops via the pybind
#     frontend; frames are returned as numpy arrays (zero-copy via DLPack).
#
# Either way, the loaded names are re-exported from this module so existing
# imports (`from torchcodec._core.ops import ...`) keep working.

import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

from torchcodec._internally_replaced_utils import (  # @manual=//pytorch/torchcodec/src:internally_replaced_utils
    load_torchcodec_shared_libraries,
)

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


expose_ffmpeg_dlls = nullcontext
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # On windows we try to locate the FFmpeg DLLs and temporarily add them to
    # the DLL search path. This seems to be needed on some users machine, but
    # not on our CI. We don't know why.
    if ffmpeg_path := shutil.which("ffmpeg"):

        def expose_ffmpeg_dlls():  # noqa: F811
            ffmpeg_dir = Path(ffmpeg_path).parent.absolute()
            return os.add_dll_directory(str(ffmpeg_dir))  # that's the actual CM


with expose_ffmpeg_dlls():
    ffmpeg_major_version, core_library_path, _pybind_ops = (
        load_torchcodec_shared_libraries()
    )


# Dispatch to the right backend and re-export all of its names (including
# underscore-prefixed ones, which `import *` would skip). The backend modules
# import the shared state (_pybind_ops, ...) from this module, which is already
# defined above by the time they're imported.
if _HAS_TORCH:
    from torchcodec._core import _torch_ops as _impl
else:
    from torchcodec._core import _numpy_ops as _impl  # type: ignore[no-redef]

globals().update(
    {key: value for key, value in vars(_impl).items() if not key.startswith("__")}
)


# The op functions are injected into this module's namespace at runtime by the
# globals().update() above, so they aren't visible to static type-checkers. Make
# them statically resolvable by re-exporting the backend surface under
# TYPE_CHECKING (this is type-check-only; runtime uses globals().update). We use
# _torch_ops as the canonical surface since it declares the full op set,
# including the underscore-prefixed ops that `import *` would skip.
if TYPE_CHECKING:
    from torchcodec._core._torch_ops import *  # noqa: F401,F403
    from torchcodec._core._torch_ops import (  # noqa: F401
        _get_backend_details,
        _get_container_json_metadata,
        _get_key_frame_indices,
        _get_log_level,
        _get_nvdec_cache_size,
        _get_stream_json_metadata,
        _set_cpp_log_level,
        _test_frame_pts_equality,
    )
