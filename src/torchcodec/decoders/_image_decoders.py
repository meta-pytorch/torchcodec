# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Private image decoders.

This module is intentionally private and minimal for now: it exposes a single
``decode_jpeg`` function that decodes a JPEG file into a ``(C, H, W)`` uint8
tensor. Only file paths are supported at the moment. More formats, input types
(bytes / tensors), a metadata probe and a unified color-conversion story will be
added later, following IMAGE_DECODER_MIGRATION_PLAN.md.
"""

import warnings
from enum import Enum
from pathlib import Path

import torch

# Importing the ops module ensures the torchcodec shared libraries (which
# register the decode_jpeg op) are loaded.
from torchcodec._core.ops import decode_jpeg as _decode_jpeg


class ImageColorMode(Enum):
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
    data = Path(path).read_bytes()
    with warnings.catch_warnings():
        # torch.frombuffer warns that the underlying buffer is non-writable;
        # we only read from the resulting tensor, so this is safe to ignore.
        warnings.filterwarnings("ignore", category=UserWarning)
        return torch.frombuffer(data, dtype=torch.uint8)


def decode_jpeg(
    source: str | Path,
    *,
    mode: ImageColorMode = ImageColorMode.UNCHANGED,
) -> torch.Tensor:
    """Decode a JPEG file into a uint8 tensor of shape ``(C, H, W)``.

    Args:
        source: Path to a JPEG file. Only file paths are supported for now.
        mode: Desired :class:`ImageColorMode`. ``UNCHANGED`` (default) keeps the
            image's native number of channels. Currently only ``UNCHANGED``,
            ``GRAY`` and ``RGB`` are supported for JPEG.

    Returns:
        A ``(C, H, W)`` uint8 tensor. EXIF orientation is always applied.
    """
    data = _read_file_to_tensor(source)
    return _decode_jpeg(data, mode.value)
