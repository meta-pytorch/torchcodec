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
    decode_avif as _decode_avif,
    decode_gif as _decode_gif,
    decode_jpeg as _decode_jpeg,
    decode_png as _decode_png,
    decode_webp as _decode_webp,
)

# TODO_IMAGE: we need to make FFmpeg an optional dependency, and probably want a
# CI job for that.

# TODO_IMAGE Some codecs expose threading options - we should make sure the
# default is 1 thread and allow the user to override it. (similar to
# num_ffmpeg_threads)

# TODO_IMAGE: I think there are some tests for corrupted images in TV? We
# should port those.

# GIF vs Pillow: our animated-GIF compositing (disposal methods, transparency,
# frame offsets) matches Pillow. GIF transparency is handled per output mode:
# - RGB/GRAY: transparent pixels are composited over the GIF background color
#   (per the GIF spec / giflib), so the output has no alpha.
# - RGB_ALPHA/GRAY_ALPHA, and UNCHANGED when the GIF has any transparent index:
#   transparency is preserved as a real alpha channel (transparent -> alpha 0,
#   uncovered/disposed regions -> fully transparent), matching Pillow and web
#   browsers, which ignore the background color for transparent GIFs.
# So UNCHANGED returns RGBA for a transparent GIF and RGB otherwise (like PNG,
# whose UNCHANGED preserves the source's native channels).


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


def _source_to_tensor(source: str | Path | bytes | torch.Tensor) -> torch.Tensor:
    # Turn any supported source into a 1-D uint8 tensor of encoded bytes.
    if isinstance(source, torch.Tensor):
        # dtype is validated in cpp.
        return source
    if isinstance(source, (str, Path)):
        # We keep the file reading in pure Python (rather than a C++ read_file
        # op like in torchvision): benchmarked against a C++ op, the
        # read is only ~1-1.5% of total decode time.
        source = Path(source).read_bytes()
    if isinstance(source, (bytes, bytearray)):
        with warnings.catch_warnings():
            # torch.frombuffer warns that the underlying buffer is non-writable;
            # we only read from the resulting tensor, so this is safe to ignore.
            warnings.filterwarnings("ignore", category=UserWarning)
            return torch.frombuffer(source, dtype=torch.uint8)

    raise TypeError(
        f"Unknown source type: {type(source)}. "
        "Supported types are str, Path, bytes and torch.Tensor."
    )


# Output modes each native codec can produce directly, for any input. Any output
# mode not listed here is obtained via a post-decode conversion.
_JPEG_NATIVE_OUTPUT_MODES = frozenset(
    (ImageColorMode.UNCHANGED, ImageColorMode.GRAY, ImageColorMode.RGB)
)
_PNG_NATIVE_OUTPUT_MODES = frozenset(ImageColorMode)
_WEBP_NATIVE_OUTPUT_MODES = _GIF_NATIVE_OUTPUT_MODES = _AVIF_NATIVE_OUTPUT_MODES = (
    frozenset((ImageColorMode.UNCHANGED, ImageColorMode.RGB, ImageColorMode.RGB_ALPHA))
)


def _append_opaque_alpha(img: torch.Tensor) -> torch.Tensor:
    # img is (..., C, H, W); append a fully-opaque alpha channel on the channel
    # dim so this works for both (C, H, W) and animated GIF (N, C, H, W) tensors.
    alpha_shape = list(img.shape)
    alpha_shape[-3] = 1
    alpha = torch.full(alpha_shape, torch.iinfo(img.dtype).max, dtype=img.dtype)
    return torch.cat([img, alpha], dim=-3)


def _rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
    # ITU-R 601-2 luma weights, matching torchvision's rgb_to_grayscale. img is
    # (..., C, H, W); reduce over the channel dim so this works for both
    # (C, H, W) and animated GIF (N, C, H, W) tensors.
    weights = torch.tensor([0.2989, 0.587, 0.114])
    gray = (img.to(torch.float32) * weights[:, None, None]).sum(dim=-3, keepdim=True)
    return gray.round().clamp(0, torch.iinfo(img.dtype).max).to(img.dtype)


def _decode_with_mode(decode_fn, data, mode, native_output_modes) -> torch.Tensor:
    if mode in native_output_modes:
        return decode_fn(data, mode.value)

    if mode is ImageColorMode.GRAY:
        # No native grayscale (e.g. webp, gif): decode RGB and reduce to luma.
        return _rgb_to_gray(decode_fn(data, ImageColorMode.RGB.value))
    elif mode is ImageColorMode.RGB_ALPHA:
        # Not native (else handled above), so the source has no real alpha:
        # synthesize an opaque one on top of RGB.
        return _append_opaque_alpha(decode_fn(data, ImageColorMode.RGB.value))
    elif mode is ImageColorMode.GRAY_ALPHA:
        if ImageColorMode.RGB_ALPHA in native_output_modes:
            # Real alpha available (e.g. webp): decode RGBA and reduce the color
            # channels to luma while preserving the alpha channel. Index the
            # channel dim (-3) so this works for both (C, H, W) and animated
            # (N, C, H, W) tensors.
            rgba = decode_fn(data, ImageColorMode.RGB_ALPHA.value)
            rgb, alpha = rgba[..., :3, :, :], rgba[..., 3:, :, :]
            return torch.cat([_rgb_to_gray(rgb), alpha], dim=-3)
        elif ImageColorMode.GRAY in native_output_modes:
            # Native gray but no alpha (e.g. jpeg): synthesize an opaque alpha.
            return _append_opaque_alpha(decode_fn(data, ImageColorMode.GRAY.value))
        else:
            # No native gray or alpha (e.g. gif): reduce RGB to luma and
            # synthesize an opaque alpha.
            gray = _rgb_to_gray(decode_fn(data, ImageColorMode.RGB.value))
            return _append_opaque_alpha(gray)
    else:
        raise RuntimeError(
            f"Reached an unexpected code path while decoding to mode {mode}. "
            "This should never happen, please report a bug to the TorchCodec repo."
        )


# TODO_IMAGE: Since we're updating the decoders code a bit, we should run sanity
# checks ensure we're not leaking anything (there was a leak on webp back then!).

# TODO_IMAGE: DOCS!! and docstrings.


def decode_jpeg(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: ImageColorMode = ImageColorMode.RGB,
) -> torch.Tensor:
    """Decode a JPEG into a uint8 tensor of shape ``(C, H, W)``.

    ``source`` can be a path (``str`` or ``pathlib.Path``), a ``bytes`` object,
    or a 1-D uint8 ``torch.Tensor`` of the raw encoded data.
    """
    data = _source_to_tensor(source)
    return _decode_with_mode(_decode_jpeg, data, mode, _JPEG_NATIVE_OUTPUT_MODES)


def decode_png(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: ImageColorMode = ImageColorMode.RGB,
) -> torch.Tensor:
    """Decode a PNG into a uint8 tensor of shape ``(C, H, W)``.

    ``source`` can be a path (``str`` or ``pathlib.Path``), a ``bytes`` object,
    or a 1-D uint8 ``torch.Tensor`` of the raw encoded data.
    """
    data = _source_to_tensor(source)
    return _decode_with_mode(_decode_png, data, mode, _PNG_NATIVE_OUTPUT_MODES)


def decode_webp(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: ImageColorMode = ImageColorMode.RGB,
) -> torch.Tensor:
    """Decode a WebP into a uint8 tensor.

    ``source`` can be a path (``str`` or ``pathlib.Path``), a ``bytes`` object,
    or a 1-D uint8 ``torch.Tensor`` of the raw encoded data.

    The shape is ``(C, H, W)`` for a still WebP and ``(N, C, H, W)`` for an
    animated one, with 4 channels when the output carries an alpha channel.
    Animated frames are composited by libwebpdemux (disposal, blending, per-frame
    offsets). The mode-conversion helpers (see _decode_with_mode) operate on the
    channel dim, so they handle both the still and animated shapes.
    """
    data = _source_to_tensor(source)
    return _decode_with_mode(_decode_webp, data, mode, _WEBP_NATIVE_OUTPUT_MODES)


def decode_gif(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: ImageColorMode = ImageColorMode.RGB,
) -> torch.Tensor:
    """Decode a GIF into a uint8 tensor.

    ``source`` can be a path (``str`` or ``pathlib.Path``), a ``bytes`` object,
    or a 1-D uint8 ``torch.Tensor`` of the raw encoded data.

    The shape is ``(C, H, W)`` for a still GIF and ``(N, C, H, W)`` for an
    animated one, with 4 channels when the output carries an alpha channel (see
    the module note on GIF transparency). The mode-conversion helpers (see
    _decode_with_mode) operate on the channel dim, so they handle both the still
    and animated shapes.
    """
    data = _source_to_tensor(source)
    return _decode_with_mode(_decode_gif, data, mode, _GIF_NATIVE_OUTPUT_MODES)


def decode_avif(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: ImageColorMode = ImageColorMode.RGB,
) -> torch.Tensor:
    # TODO_IMAGE: 10/12-bit AVIF files are decoded as uint8 for now, losing
    # precision. Returning uint16 for high-bit-depth files is tied to adding
    """Decode an AVIF into a uint8 tensor.

    ``source`` can be a path (``str`` or ``pathlib.Path``), a ``bytes`` object,
    or a 1-D uint8 ``torch.Tensor`` of the raw encoded data.
    """
    data = _source_to_tensor(source)
    return _decode_with_mode(_decode_avif, data, mode, _AVIF_NATIVE_OUTPUT_MODES)
