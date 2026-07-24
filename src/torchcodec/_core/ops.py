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
    load_core_libraries,
    load_heic_library,
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

decode_jpeg = torch.ops.torchcodec_ns.decode_jpeg.default
decode_png = torch.ops.torchcodec_ns.decode_png.default
decode_webp = torch.ops.torchcodec_ns.decode_webp.default
decode_gif = torch.ops.torchcodec_ns.decode_gif.default
decode_avif = torch.ops.torchcodec_ns.decode_avif.default


def get_decode_heic():
    # decode_heic lives in the separate, lazily-loaded libtorchcodec_heic (see
    # load_heic_library), so unlike the decoders above we can't bind its op at
    # import time: the op isn't registered until the library is loaded. This
    # helper loads the library (raising an actionable ImportError if libheif
    # isn't available) and returns the op.
    load_heic_library()
    return torch.ops.torchcodec_ns.decode_heic.default


# FFmpeg is now an optional dependency, since the image decoders above don't
# need it, and we want users to be able to use them without having FFmpeg
# installed.  We try to load the ffmpeg-dependent ops at import time and bind
# their names to the actual implementation. If it doesn't work
# (_FFMPEG_AVAILABLE is then False), we bind the names to a dummy lambda that
# will raise a proper error when called.
try:
    with expose_ffmpeg_dlls():
        ffmpeg_major_version, core_library_path, _ = load_core_libraries()
    _FFMPEG_AVAILABLE = True
except Exception:
    _FFMPEG_AVAILABLE = False
    ffmpeg_major_version = None
    core_library_path = None


if _FFMPEG_AVAILABLE:
    from torchcodec._core._ffmpeg_ops import *  # noqa: F401,F403
else:

    def __getattr__(name):
        # Mocks all names in FFMPEG_OP_NAMES to raise an error when called, if
        # FFmpeg is not available.
        if name in FFMPEG_OP_NAMES:
            return lambda *args, **kwargs: load_core_libraries()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
