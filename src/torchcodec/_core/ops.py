# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torchcodec._core._ffmpeg_op_names import FFMPEG_OP_NAMES
from torchcodec._internally_replaced_utils import (  # @manual=//pytorch/torchcodec/src:internally_replaced_utils
    ensure_ffmpeg_loaded,
    load_image_library,
)


expose_ffmpeg_dlls = nullcontext
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # On windows we try to locate the FFmpeg DLLs and temporarily add them to
    # the DLL search path. This seems to be needed on some users machine, but
    # not on our CI. We don't know why.
    if ffmpeg_path := shutil.which("ffmpeg"):

        def expose_ffmpeg_dlls():  # noqa: F811
            ffmpeg_dir = Path(ffmpeg_path).parent.absolute()
            return os.add_dll_directory(str(ffmpeg_dir))  # that's the actual CM


load_image_library()

# Image ops live in the FFmpeg-free libtorchcodec_image library, so they are
# always available, even when FFmpeg is not installed.
decode_jpeg = torch.ops.torchcodec_ns.decode_jpeg.default
decode_png = torch.ops.torchcodec_ns.decode_png.default
decode_webp = torch.ops.torchcodec_ns.decode_webp.default
decode_gif = torch.ops.torchcodec_ns.decode_gif.default
decode_avif = torch.ops.torchcodec_ns.decode_avif.default

# FFmpeg is an optional dependency: try to load it eagerly (so the real ops are
# bound and torch.compile works when it's present), but tolerate its absence.
try:
    with expose_ffmpeg_dlls():
        ffmpeg_major_version, core_library_path, _ = ensure_ffmpeg_loaded()
    _FFMPEG_AVAILABLE = True
except Exception:
    _FFMPEG_AVAILABLE = False
    ffmpeg_major_version = None
    core_library_path = None


if _FFMPEG_AVAILABLE:
    # The real ops, wrappers and torch.compile fakes live in _ffmpeg_ops so this
    # file stays flat; `import *` re-exports them here (its __all__ lists every
    # name, including the underscore-prefixed ones).
    from torchcodec._core._ffmpeg_ops import *  # noqa: F401,F403
else:

    def __getattr__(name):
        # FFmpeg is unavailable, so the FFmpeg-backed ops/helpers weren't bound.
        # Resolve them (and only them) to a stub that raises the rich, actionable
        # error when called: this keeps `import torchcodec` and
        # `from torchcodec._core.ops import <op>` working, and only fails when an
        # FFmpeg-backed feature is actually used. Everything else (typos,
        # introspection) raises AttributeError as usual.
        if name in FFMPEG_OP_NAMES:
            return lambda *args, **kwargs: ensure_ffmpeg_loaded()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
