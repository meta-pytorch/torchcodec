# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from enum import Enum
from pathlib import Path

import torch

from torchcodec._core.ops import (
    decode_jpeg as _decode_jpeg,
    decode_png as _decode_png,
    decode_webp as _decode_webp,
)

# TODO_IMAGE: we need to make FFmpeg an optional dependency

# TODO_IMAGE: we should allow to build without all the image dependencies
# (libjpeg etc.), and make sure only the *calls* to decode_*() fail at runtime,
# not the import of torchcodec.

# TODO_IMAGE We probably need CI jobs for both TODOs above.

# TODO_IMAGE: Support torchscript?

# TODO_IMAGE: We'll need to support all output modes consistently across all
# decoders, with tests.

# TODO_IMAGE: I think there are some tests for corrupted images in TV? We
# should port those.


class ImageColorMode(Enum):
    # TODO_IMAGE:  We'll probably need to keep that for BC but ugh. this should
    # be ImageReadMode to be consistent with TV. Let's type stuff with Literal
    # strings instead.

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


# Output modes each native codec can produce directly, for any input. Any output
# mode not listed here is obtained via a post-decode conversion.
_JPEG_NATIVE_OUTPUT_MODES = frozenset(
    (ImageColorMode.UNCHANGED, ImageColorMode.GRAY, ImageColorMode.RGB)
)
_PNG_NATIVE_OUTPUT_MODES = frozenset(ImageColorMode)
_WEBP_NATIVE_OUTPUT_MODES = frozenset(
    (ImageColorMode.UNCHANGED, ImageColorMode.RGB, ImageColorMode.RGB_ALPHA)
)


def _append_opaque_alpha(img: torch.Tensor) -> torch.Tensor:
    _, height, width = img.shape
    alpha = torch.full((1, height, width), torch.iinfo(img.dtype).max, dtype=img.dtype)
    return torch.cat([img, alpha], dim=0)


def _rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
    # ITU-R 601-2 luma weights, matching torchvision's rgb_to_grayscale.
    weights = torch.tensor([0.2989, 0.587, 0.114])
    gray = (img.to(torch.float32) * weights[:, None, None]).sum(dim=0, keepdim=True)
    return gray.round().clamp(0, torch.iinfo(img.dtype).max).to(img.dtype)


def _decode_with_mode(decode_fn, data, mode, native_output_modes) -> torch.Tensor:
    if mode in native_output_modes:
        return decode_fn(data, mode.value)
    if mode is ImageColorMode.GRAY:
        # Reached for decoders without native grayscale support (e.g. webp):
        # decode RGB and reduce to luma.
        return _rgb_to_gray(decode_fn(data, ImageColorMode.RGB.value))
    if mode is ImageColorMode.GRAY_ALPHA:
        if ImageColorMode.GRAY in native_output_modes:
            # The decoder produces gray but not gray+alpha (e.g. jpeg): the
            # source carries no alpha, so synthesize an opaque one.
            return _append_opaque_alpha(decode_fn(data, ImageColorMode.GRAY.value))
        else:
            # No native grayscale (e.g. webp): decode RGBA and reduce color to
            # luma while preserving the real alpha channel.
            rgba = decode_fn(data, ImageColorMode.RGB_ALPHA.value)
            return torch.cat([_rgb_to_gray(rgba[:3]), rgba[3:]], dim=0)
    if mode is ImageColorMode.RGB_ALPHA:
        return _append_opaque_alpha(decode_fn(data, ImageColorMode.RGB.value))
    raise RuntimeError(
        f"Reached an unexpected code path while decoding to mode {mode}. "
        "This should never happen, please report a bug to the TorchCodec repo."
    )


# TODO_IMAGE: Since we're updating the decoders code a bit, we should run sanity
# checks ensure we're not leaking anything (there was a leak on webp back then!).


def decode_jpeg(
    # TODO_IMAGE: support bytes and file-like
    source: str | Path,
    *,
    mode: ImageColorMode = ImageColorMode.RGB,
) -> torch.Tensor:
    # TODO_IMAGE We should ensure we build and link against turbo. Maybe by
    # checking the symbols of the bundled libjpeg shared library at repair time.
    """Decode a JPEG file into a uint8 tensor of shape ``(C, H, W)``."""
    data = _read_file_to_tensor(source)
    return _decode_with_mode(_decode_jpeg, data, mode, _JPEG_NATIVE_OUTPUT_MODES)


def decode_png(
    source: str | Path,
    *,
    mode: ImageColorMode = ImageColorMode.RGB,
) -> torch.Tensor:
    """Decode a PNG file into a uint8 tensor of shape ``(C, H, W)``."""
    data = _read_file_to_tensor(source)
    return _decode_with_mode(_decode_png, data, mode, _PNG_NATIVE_OUTPUT_MODES)


def decode_webp(
    source: str | Path,
    *,
    mode: ImageColorMode = ImageColorMode.RGB,
) -> torch.Tensor:
    # TODO_IMAGE: animated webp files are not supported yet.
    """Decode a WebP file into a uint8 tensor of shape ``(C, H, W)``."""
    data = _read_file_to_tensor(source)
    return _decode_with_mode(_decode_webp, data, mode, _WEBP_NATIVE_OUTPUT_MODES)
