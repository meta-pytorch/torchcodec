# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import torch

from torchcodec._core.ops import (
    decode_avif as _decode_avif,
    decode_gif as _decode_gif,
    decode_jpeg as _decode_jpeg,
    decode_jpegs_cuda as _decode_jpegs_cuda,
    decode_png as _decode_png,
    decode_webp as _decode_webp,
)

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


class ImageReadMode(Enum):
    """Color mode for image decoding, mirroring torchvision's ``ImageReadMode``.

    The recommended way to specify a color mode is a (case-insensitive) string
    such as ``"rgb"``; this enum is only kept for backward compatibility and is
    accepted anywhere a mode string is. Its integer values match torchvision's
    ``ImageReadMode`` and the C++ ``ImageReadMode`` constants.
    """

    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4


def _normalize_mode(
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ),
) -> ImageReadMode:
    # Normalize the public `mode` argument (a case-insensitive string, or an
    # ImageReadMode for BC) to an ImageReadMode, which is what the rest of the
    # decoding code works with.
    if isinstance(mode, ImageReadMode):
        return mode
    if isinstance(mode, str):
        try:
            return ImageReadMode[mode.upper()]
        except KeyError:
            valid = ", ".join(repr(m.name) for m in ImageReadMode)
            raise ValueError(
                f"Invalid mode {mode!r}. Supported modes are {valid} "
                "(case-insensitive)."
            ) from None
    raise TypeError(f"mode must be a str (or ImageReadMode), got {type(mode)}.")


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
    (ImageReadMode.UNCHANGED, ImageReadMode.GRAY, ImageReadMode.RGB)
)
_PNG_NATIVE_OUTPUT_MODES = frozenset(ImageReadMode)
_WEBP_NATIVE_OUTPUT_MODES = _GIF_NATIVE_OUTPUT_MODES = _AVIF_NATIVE_OUTPUT_MODES = (
    frozenset((ImageReadMode.UNCHANGED, ImageReadMode.RGB, ImageReadMode.RGB_ALPHA))
)


def _append_opaque_alpha(img: torch.Tensor) -> torch.Tensor:
    # img is (..., C, H, W); append a fully-opaque alpha channel on the channel
    # dim so this works for both (C, H, W) and animated GIF (N, C, H, W) tensors.
    # Keep the alpha on img's device so this also works for CUDA-decoded JPEGs.
    alpha_shape = list(img.shape)
    alpha_shape[-3] = 1
    alpha = torch.full(
        alpha_shape, torch.iinfo(img.dtype).max, dtype=img.dtype, device=img.device
    )
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

    if mode is ImageReadMode.GRAY:
        # No native grayscale (e.g. webp, gif): decode RGB and reduce to luma.
        return _rgb_to_gray(decode_fn(data, ImageReadMode.RGB.value))
    elif mode is ImageReadMode.RGB_ALPHA:
        # Not native (else handled above), so the source has no real alpha:
        # synthesize an opaque one on top of RGB.
        return _append_opaque_alpha(decode_fn(data, ImageReadMode.RGB.value))
    elif mode is ImageReadMode.GRAY_ALPHA:
        if ImageReadMode.RGB_ALPHA in native_output_modes:
            # Real alpha available (e.g. webp): decode RGBA and reduce the color
            # channels to luma while preserving the alpha channel. Index the
            # channel dim (-3) so this works for both (C, H, W) and animated
            # (N, C, H, W) tensors.
            rgba = decode_fn(data, ImageReadMode.RGB_ALPHA.value)
            rgb, alpha = rgba[..., :3, :, :], rgba[..., 3:, :, :]
            return torch.cat([_rgb_to_gray(rgb), alpha], dim=-3)
        elif ImageReadMode.GRAY in native_output_modes:
            # Native gray but no alpha (e.g. jpeg): synthesize an opaque alpha.
            return _append_opaque_alpha(decode_fn(data, ImageReadMode.GRAY.value))
        else:
            # No native gray or alpha (e.g. gif): reduce RGB to luma and
            # synthesize an opaque alpha.
            gray = _rgb_to_gray(decode_fn(data, ImageReadMode.RGB.value))
            return _append_opaque_alpha(gray)
    else:
        raise RuntimeError(
            f"Reached an unexpected code path while decoding to mode {mode}. "
            "This should never happen, please report a bug to the TorchCodec repo."
        )


# Maps the output_dtype API values to the integer codes understood by the C++
# decoders. Must be kept in-sync with the OutputDtype enum in ImageCommon.h.
_OUTPUT_DTYPE_TO_CODE = {torch.uint8: 0, torch.uint16: 1, "auto": 2}


def _validate_output_dtype(output_dtype) -> None:
    if output_dtype not in _OUTPUT_DTYPE_TO_CODE:
        raise ValueError(
            f"Invalid output_dtype ({output_dtype}). "
            "Supported values are torch.uint8, torch.uint16, and 'auto'."
        )


def _maybe_widen_to_uint16(
    decoded: torch.Tensor, output_dtype: torch.dtype | str
) -> torch.Tensor:
    # For the always-8-bit codecs, the codec cannot emit 16-bit samples, so
    # uint16 output is produced here by scaling the 8-bit values up to the full
    # 16-bit range (a factor of 257 == 65535 / 255, so 255 -> 65535).
    if decoded.dtype != torch.uint8:
        raise RuntimeError("should never happen, please report a bug")
    if output_dtype == torch.uint16:
        return (decoded.to(torch.int32) * 257).to(torch.uint16)
    else:
        return decoded


# TODO_IMAGE: Since we're updating the decoders code a bit, we should run sanity
# checks ensure we're not leaking anything (there was a leak on webp back then!).

# TODO_IMAGE: DOCS!! and docstrings.


# Shared semantics of ``output_dtype`` for all decoders below: ``torch.uint8``
# (the default) always yields an 8-bit tensor and ``torch.uint16`` always a
# 16-bit one, rescaling to the full range of the target dtype as needed.
# ``"auto"`` keeps the source's native precision: uint8 for 8-bit sources and
# uint16 for sources carrying more than 8 bits per channel (16-bit PNG, 10/12-bit
# AVIF). JPEG, WebP and GIF are always 8-bit, so for them "auto" is equivalent to
# torch.uint8 and torch.uint16 simply widens the 8-bit values.


def _decode_jpegs_cuda_with_mode(
    tensors: list[torch.Tensor], mode: ImageReadMode, device: torch.device
) -> list[torch.Tensor]:
    # Batched GPU equivalent of _decode_with_mode for JPEG: decode the whole
    # batch in one nvJPEG call using a native mode, then emulate the alpha modes
    # per-image in Python (nvJPEG natively supports UNCHANGED, GRAY and RGB, same
    # as libjpeg on CPU).
    if mode in _JPEG_NATIVE_OUTPUT_MODES:
        return _decode_jpegs_cuda(tensors, mode.value, device)
    if mode is ImageReadMode.RGB_ALPHA:
        decoded = _decode_jpegs_cuda(tensors, ImageReadMode.RGB.value, device)
        return [_append_opaque_alpha(img) for img in decoded]
    if mode is ImageReadMode.GRAY_ALPHA:
        decoded = _decode_jpegs_cuda(tensors, ImageReadMode.GRAY.value, device)
        return [_append_opaque_alpha(img) for img in decoded]
    raise RuntimeError(
        f"Reached an unexpected code path while decoding to mode {mode}. "
        "This should never happen, please report a bug to the TorchCodec repo."
    )


def decode_jpeg(
    source: str | Path | bytes | torch.Tensor | list,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
    device: str | torch.device = "cpu",
) -> torch.Tensor | list[torch.Tensor]:
    """Decode a JPEG into a tensor of shape ``(C, H, W)``.

    ``source`` can be a path (``str`` or ``pathlib.Path``), a ``bytes`` object,
    or a 1-D uint8 ``torch.Tensor`` of the raw encoded data. ``mode`` is a
    case-insensitive color mode string (e.g. ``"rgb"``, ``"gray"``). See the
    module note above for the semantics of ``output_dtype``.

    ``device`` selects where decoding happens: ``"cpu"`` (the default) uses
    libjpeg-turbo, while a CUDA device (e.g. ``"cuda"``) decodes on the GPU with
    nvJPEG and returns tensors on that device. On CUDA, ``source`` may also be a
    list of sources, in which case a list of ``(C, H, W)`` tensors (one per
    input) is returned and the batch is decoded together for higher throughput.
    """
    _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    device = torch.device(device)

    if device.type == "cpu":
        if isinstance(source, (list, tuple)):
            raise ValueError(
                "Batch decoding (a list of sources) is only supported on CUDA "
                "devices. Pass device='cuda' or decode sources one at a time."
            )
        data = _source_to_tensor(source)
        decoded = _decode_with_mode(_decode_jpeg, data, mode, _JPEG_NATIVE_OUTPUT_MODES)
        return _maybe_widen_to_uint16(decoded, output_dtype)

    if isinstance(source, (list, tuple)):
        is_batch = True
        sources: list = list(source)
    else:
        is_batch = False
        sources = [source]
    tensors = [_source_to_tensor(s) for s in sources]
    decoded_list = _decode_jpegs_cuda_with_mode(tensors, mode, device)
    decoded_list = [_maybe_widen_to_uint16(img, output_dtype) for img in decoded_list]
    return decoded_list if is_batch else decoded_list[0]


def decode_png(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode a PNG into a tensor of shape ``(C, H, W)``.

    ``source`` can be a path (``str`` or ``pathlib.Path``), a ``bytes`` object,
    or a 1-D uint8 ``torch.Tensor`` of the raw encoded data. ``mode`` is a
    case-insensitive color mode string (e.g. ``"rgb"``, ``"gray"``). See the
    module note above for the semantics of ``output_dtype``.
    """
    _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    code = _OUTPUT_DTYPE_TO_CODE[output_dtype]
    return _decode_with_mode(
        lambda d, m: _decode_png(d, m, code), data, mode, _PNG_NATIVE_OUTPUT_MODES
    )


def decode_webp(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode a WebP into a tensor.

    ``source`` can be a path (``str`` or ``pathlib.Path``), a ``bytes`` object,
    or a 1-D uint8 ``torch.Tensor`` of the raw encoded data. ``mode`` is a
    case-insensitive color mode string (e.g. ``"rgb"``, ``"gray"``).

    The shape is ``(C, H, W)`` for a still WebP and ``(N, C, H, W)`` for an
    animated one, with 4 channels when the output carries an alpha channel.
    Animated frames are composited by libwebpdemux (disposal, blending, per-frame
    offsets). The mode-conversion helpers (see _decode_with_mode) operate on the
    channel dim, so they handle both the still and animated shapes.

    See the module note above for the semantics of ``output_dtype``.
    """
    _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    decoded = _decode_with_mode(_decode_webp, data, mode, _WEBP_NATIVE_OUTPUT_MODES)
    return _maybe_widen_to_uint16(decoded, output_dtype)


def decode_gif(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode a GIF into a tensor.

    ``source`` can be a path (``str`` or ``pathlib.Path``), a ``bytes`` object,
    or a 1-D uint8 ``torch.Tensor`` of the raw encoded data. ``mode`` is a
    case-insensitive color mode string (e.g. ``"rgb"``, ``"gray"``).

    The shape is ``(C, H, W)`` for a still GIF and ``(N, C, H, W)`` for an
    animated one, with 4 channels when the output carries an alpha channel (see
    the module note on GIF transparency). The mode-conversion helpers (see
    _decode_with_mode) operate on the channel dim, so they handle both the still
    and animated shapes.

    See the module note above for the semantics of ``output_dtype``.
    """
    _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    decoded = _decode_with_mode(_decode_gif, data, mode, _GIF_NATIVE_OUTPUT_MODES)
    return _maybe_widen_to_uint16(decoded, output_dtype)


def decode_avif(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
    num_threads: int = 1,
) -> torch.Tensor:
    """Decode an AVIF into a tensor of shape ``(C, H, W)``.

    ``source`` can be a path (``str`` or ``pathlib.Path``), a ``bytes`` object,
    or a 1-D uint8 ``torch.Tensor`` of the raw encoded data. ``mode`` is a
    case-insensitive color mode string (e.g. ``"rgb"``, ``"gray"``). See the
    module note above for the semantics of ``output_dtype``; 10- and 12-bit AVIF
    sources carry more than 8 bits per channel, so ``"auto"`` and
    ``torch.uint16`` preserve that precision.
    """
    _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    code = _OUTPUT_DTYPE_TO_CODE[output_dtype]
    return _decode_with_mode(
        lambda d, m: _decode_avif(d, m, code, num_threads),
        data,
        mode,
        _AVIF_NATIVE_OUTPUT_MODES,
    )


# Maps a detected format to its public decoder, so decode_image reuses the exact
# same decoding path (mode emulation and output_dtype handling) as the
# format-specific decoders above.
# decode_jpeg's return type (Tensor | list[Tensor]) differs from the others, so
# we type the values as Callable[..., Any] to keep the dict callable for mypy.
# decode_image only ever calls these with a single source (no device), so each
# returns a plain Tensor at runtime.
_FORMAT_TO_DECODER: dict[str, Callable[..., Any]] = {
    "jpeg": decode_jpeg,
    "png": decode_png,
    "webp": decode_webp,
    "gif": decode_gif,
    "avif": decode_avif,
}


def _detect_image_format(data: torch.Tensor) -> str:
    # Sniff the codec from the leading "magic" bytes of the encoded data.
    # This used to be implemented in C++ in torchvision, but benchmarks show
    # this is negligible in Python
    header = bytes(data[:64].tolist())
    if header[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    if header[4:8] == b"ftyp":
        # ISOBMFF container (AVIF/HEIC/...). The major brand is at [8:12], with
        # compatible brands following. AVIF uses the "avif" (still) or "avis"
        # (animated) brands; HEIC (which we don't support) uses "heic"/"heix"
        # and is deliberately not matched here.
        brands = header[8:]
        if b"avif" in brands or b"avis" in brands:
            return "avif"
    raise ValueError(
        "Unsupported or unrecognized image format. Supported formats are "
        "JPEG, PNG, WebP, GIF and AVIF. If you know you have a valid image, "
        "try using the dedicated decode_* functions like decode_jpeg() instead."
    )


# Design note: the parameters of decode_image must apply to *all* codecs
# uniformly. That's why all modes are supported by decode_image even though not
# all codec would natively support all mode - e.g. jpeg has no alpha support, so
# we prepend an opaque alpha channel as a post-processing step.  As a resut, all
# codec-specific entry points like decode_jpeg, decode_png etc.  must still
# expose the same parameters that decode_image exposes.  The codec-specific
# parameters should live in the codec-specific entry points, e.g. decode_avif
# has its `num_threads`, decode_jpeg has `device`, etc.
def decode_image(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode an image into a tensor, detecting the format automatically.

    ``source`` can be a path (``str`` or ``pathlib.Path``), a ``bytes`` object,
    or a 1-D uint8 ``torch.Tensor`` of the raw encoded data. ``mode`` is a
    case-insensitive color mode string (e.g. ``"rgb"``, ``"gray"``). See the
    module note above for the semantics of ``output_dtype``.
    """
    _validate_output_dtype(output_dtype)
    data = _source_to_tensor(source)
    fmt = _detect_image_format(data)
    return _FORMAT_TO_DECODER[fmt](data, mode=mode, output_dtype=output_dtype)
