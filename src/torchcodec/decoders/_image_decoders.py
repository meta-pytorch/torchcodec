# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from enum import Enum
from pathlib import Path

import torch

from torchcodec._core.ops import decode_jpeg as _decode_jpeg

# TODO_IMAGE: we need to make FFmpeg an optional dependency

# TODO_IMAGE: we should allow to build without all the image dependencies
# (libjpeg etc.), and make sure only the *calls* to decode_*() fail at runtime,
# not the import of torchcodec.

# TODO_IMAGE We probably need CI jobs for both TODOs above.

# TODO_IMAGE: Support torchscript?

# TODO_IMAGE: We'll need to support all output modes consistently across all
# decoders, with tests.


class ImageColorMode(Enum):
    # TODO_IMAGE:  We'll probably need to keep that for BC but ugh. Let's type
    # stuff with Literal strings instead.
    """Color mode for image decoding.

    Integer values match torchvision's ``ImageReadMode`` and the C++
    ``ImageReadMode`` constants.
    """

    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4


def _read_file_to_tensor(path: str | Path) -> torch.Tensor:
    # TODO_IMAGE: port read_file?
    data = Path(path).read_bytes()
    with warnings.catch_warnings():
        # torch.frombuffer warns that the underlying buffer is non-writable;
        # we only read from the resulting tensor, so this is safe to ignore.
        warnings.filterwarnings("ignore", category=UserWarning)
        return torch.frombuffer(data, dtype=torch.uint8)


def decode_jpeg(
    # TODO_IMAGE: support bytes and file-like
    source: str | Path,
    *,
    mode: ImageColorMode = ImageColorMode.UNCHANGED,
) -> torch.Tensor:
    # TODO_IMAGE We should ensure we build and link against turbo. Maybe by
    # checking the symbols of the bundled libjpeg shared library at repair time.
    """Decode a JPEG file into a uint8 tensor of shape ``(C, H, W)``.

    Args:
        source: Path to a JPEG file. Only file paths are supported for now.
        mode: Desired :class:`ImageColorMode`. ``UNCHANGED`` (default) keeps the
            image's native number of channels. Currently only ``UNCHANGED``,
            ``GRAY`` and ``RGB`` are supported for JPEG.

    Returns:
        A ``(C, H, W)`` uint8 tensor.
    """
    data = _read_file_to_tensor(source)
    return _decode_jpeg(data, mode.value)
