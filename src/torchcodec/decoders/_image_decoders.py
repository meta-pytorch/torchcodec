# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import struct
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from torchcodec._core.ops import (
    decode_avif as _decode_avif,
    decode_gif as _decode_gif,
    decode_jpeg as _decode_jpeg,
    decode_jpegs_cuda as _decode_jpegs_cuda,
    decode_png as _decode_png,
    decode_webp as _decode_webp,
    get_decode_heic as _get_decode_heic,
)


class ImageReadMode(Enum):
    """Color mode for image decoding.

    You don't have to use this, you can just pass strings like "RGB" or "GRAY" instead.
    """

    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4
    RGBA = 4  # undocumented alias for RGB_ALPHA


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
    _HEIC_NATIVE_OUTPUT_MODES
) = frozenset((ImageReadMode.UNCHANGED, ImageReadMode.RGB, ImageReadMode.RGB_ALPHA))


def _append_opaque_alpha(img: torch.Tensor) -> torch.Tensor:
    # Append a fully-opaque alpha channel on the channel dim
    # works on CHW and NCHW tensors.
    alpha_shape = list(img.shape)
    alpha_shape[-3] = 1
    alpha = torch.full(
        alpha_shape, torch.iinfo(img.dtype).max, dtype=img.dtype, device=img.device
    )
    return torch.cat([img, alpha], dim=-3)


def _rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
    # ITU-R 601-2 luma weights, matching torchvision's rgb_to_grayscale.
    # works on CHW and NCHW tensors.
    weights = torch.tensor([0.2989, 0.587, 0.114])
    gray = (img.to(torch.float32) * weights[:, None, None]).sum(dim=-3, keepdim=True)
    return gray.round().clamp(0, torch.iinfo(img.dtype).max).to(img.dtype)


def _decode_to_mode(decode_fn, data, mode, native_output_modes) -> torch.Tensor:
    if mode in native_output_modes:
        return decode_fn(data, mode.value)

    if mode is ImageReadMode.GRAY:
        return _rgb_to_gray(decode_fn(data, ImageReadMode.RGB.value))
    elif mode is ImageReadMode.RGB_ALPHA:
        return _append_opaque_alpha(decode_fn(data, ImageReadMode.RGB.value))
    elif mode is ImageReadMode.GRAY_ALPHA:
        if ImageReadMode.RGB_ALPHA in native_output_modes:
            # Real alpha available (e.g. webp): decode RGBA and reduce the color
            # channels to luma while preserving the alpha channel.
            rgba = decode_fn(data, ImageReadMode.RGB_ALPHA.value)
            rgb, alpha = rgba[..., :3, :, :], rgba[..., 3:, :, :]
            return torch.cat([_rgb_to_gray(rgb), alpha], dim=-3)
        elif ImageReadMode.GRAY in native_output_modes:
            return _append_opaque_alpha(decode_fn(data, ImageReadMode.GRAY.value))
        else:
            gray = _rgb_to_gray(decode_fn(data, ImageReadMode.RGB.value))
            return _append_opaque_alpha(gray)
    else:
        raise RuntimeError(
            f"Reached an unexpected code path while decoding to mode {mode}. "
            "This should never happen, please report a bug to the TorchCodec repo."
        )


def _validate_output_dtype(output_dtype) -> int:
    # Validates output_dtype and returns the integer code understood by the C++
    # decoders. Must be kept in-sync with the OutputDtype enum in ImageCommon.h.
    output_dtype_to_code = {torch.uint8: 0, torch.uint16: 1, "auto": 2}
    if output_dtype not in output_dtype_to_code:
        raise ValueError(
            f"Invalid output_dtype ({output_dtype}). "
            "Supported values are torch.uint8, torch.uint16, and 'auto'."
        )
    return output_dtype_to_code[output_dtype]


def _to_output_dtype(
    decoded: torch.Tensor, output_dtype: torch.dtype | str
) -> torch.Tensor:
    if output_dtype == "auto" or decoded.dtype == output_dtype:
        return decoded
    elif output_dtype == torch.uint16:
        if decoded.dtype != torch.uint8:
            raise RuntimeError("Oops, please report a bug to the TorchCodec repo.")
        return (decoded.to(torch.int32) * 257).to(torch.uint16)
    elif output_dtype == torch.uint8:
        if decoded.dtype != torch.uint16:
            raise RuntimeError("Oops, please report a bug to the TorchCodec repo.")
        return (decoded.to(torch.float32) / 257).round().clamp(0, 255).to(torch.uint8)
    else:
        raise RuntimeError(
            "This should never happen, please report a bug to the TorchCodec repo."
        )


def _decode_jpegs_cuda_with_mode(
    tensors: list[torch.Tensor], mode: ImageReadMode, device: torch.device
) -> list[torch.Tensor]:
    # Batched GPU equivalent of _decode_to_mode for JPEG: decode the whole
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
    """Decode a JPEG image into a ``CHW`` tensor, on CPU or CUDA.

    .. note::

        For CUDA decoding, prefer passing a batch (a list of sources) in a
        single call: the whole batch is decoded in one nvJPEG call, which is
        much faster than decoding images one at a time.
        Passing a batch of sources is supported on CPU too, but it won't be
        faster than decoding them one at a time.

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_jpeg

            img = decode_jpeg("image.jpg")
            img = decode_jpeg("image.jpg", device="cuda")  # decode on GPU

    Args:
        source (str, ``pathlib.Path``, bytes, ``torch.Tensor``, or list of these):
            The encoded JPEG data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes. Pass a list (or tuple) to decode a batch, in which case a list of
            tensors is returned instead of a single tensor. The encoded bytes must
            live on CPU, even when decoding to a CUDA device.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. Since JPEG is an 8-bit format,
            ``"auto"`` and ``torch.uint8`` are equivalent. ``torch.uint16``
            emulates a 16-bit output by scaling the 8-bit values to the full
            16-bit range (0-255 -> 0-65535).
        device (str or torch.device, optional): Device to decode on, ``"cpu"``
            (default) or a CUDA device. CUDA decoding uses nvJPEG. We recommend
            passing a batch of sources when decoding on CUDA, for speed.

    Returns:
        torch.Tensor or list of torch.Tensor of shape ``C, H, W``: The decoded
        image(s). A single tensor for a single source, or a list of tensors for
        a batch.
    """
    _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    device = torch.device(device)

    is_batch = isinstance(source, (list, tuple))
    sources: list = list(source) if is_batch else [source]  # type: ignore[arg-type]

    if device.type == "cpu":
        decoded_list = [
            _to_output_dtype(
                _decode_to_mode(
                    _decode_jpeg,
                    _source_to_tensor(s),
                    mode,
                    _JPEG_NATIVE_OUTPUT_MODES,
                ),
                output_dtype,
            )
            for s in sources
        ]
    else:
        tensors = [_source_to_tensor(s) for s in sources]
        decoded_list = _decode_jpegs_cuda_with_mode(tensors, mode, device)
        decoded_list = [_to_output_dtype(img, output_dtype) for img in decoded_list]

    return decoded_list if is_batch else decoded_list[0]


def decode_png(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode a PNG image into a ``CHW`` tensor.

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_png

            img = decode_png("image.png")

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded PNG data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. PNG images can natively store
            16-bit samples: ``torch.uint16`` preserves that precision (8-bit
            sources are scaled up, 0-255 -> 0-65535), while ``torch.uint8``
            scales 16-bit sources down. ``"auto"`` keeps the source's native bit
            depth, yielding uint8 for 8-bit PNGs and uint16 for 16-bit ones.

    Returns:
        torch.Tensor: The decoded image, of shape ``(C, H, W)``.
    """
    output_dtype_code = _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    return _decode_to_mode(
        lambda d, m: _decode_png(d, m, output_dtype_code),
        data,
        mode,
        _PNG_NATIVE_OUTPUT_MODES,
    )


def decode_webp(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode a WebP image into a ``[N]CHW`` tensor.

    The output shape is ``(C, H, W)`` for a still WebP and ``(N, C, H, W)`` for
    an animated one (N frames).

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_webp

            img = decode_webp("image.webp")

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded WebP data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. Since WebP is an 8-bit format,
            ``"auto"`` and ``torch.uint8`` are equivalent. ``torch.uint16``
            emulates a 16-bit output by scaling the 8-bit values to the full
            16-bit range (0-255 -> 0-65535).

    Returns:
        torch.Tensor: The decoded image, of shape ``(C, H, W)`` (still) or
        ``(N, C, H, W)`` (animated).
    """
    _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    decoded = _decode_to_mode(_decode_webp, data, mode, _WEBP_NATIVE_OUTPUT_MODES)
    return _to_output_dtype(decoded, output_dtype)


def decode_gif(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode a GIF image into a ``[N]CHW`` tensor.

    The output shape is ``(C, H, W)`` for a still GIF and ``(N, C, H, W)`` for
    an animated one (N frames).

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_gif

            img = decode_gif("image.gif")

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded GIF data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. Since GIF is an 8-bit format,
            ``"auto"`` and ``torch.uint8`` are equivalent. ``torch.uint16``
            emulates a 16-bit output by scaling the 8-bit values to the full
            16-bit range (0-255 -> 0-65535).

    Returns:
        torch.Tensor: The decoded image, of shape ``(C, H, W)`` (still) or
        ``(N, C, H, W)`` (animated).
    """
    _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    decoded = _decode_to_mode(_decode_gif, data, mode, _GIF_NATIVE_OUTPUT_MODES)
    return _to_output_dtype(decoded, output_dtype)


def decode_avif(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
    num_threads: int = 1,
) -> torch.Tensor:
    """Decode an AVIF image into a ``[N]CHW`` tensor.

    The output shape is ``(C, H, W)`` for a still AVIF and ``(N, C, H, W)`` for
    an animated one (N frames).

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_avif

            img = decode_avif("image.avif")

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded AVIF data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. AVIF can store more than 8 bits per
            channel (e.g. 10- or 12-bit sources). ``torch.uint16`` always scales
            the samples up to fill the full 16-bit range ``[0, 65535]`` (8-bit
            0-255, 10-bit 0-1023 and 12-bit 0-4095 sources are all upscaled),
            while ``torch.uint8`` scales higher-bit sources down. ``"auto"``
            yields uint8 for 8-bit AVIFs and uint16 (again filling ``[0, 65535]``)
            for higher-bit ones.
        num_threads (int, optional): Number of threads to use for decoding,
            directly passed to libavif. Default is 1.

    Returns:
        torch.Tensor: The decoded image, of shape ``(C, H, W)`` (still) or
        ``(N, C, H, W)`` (animated).
    """
    output_dtype_code = _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    return _decode_to_mode(
        lambda d, m: _decode_avif(d, m, output_dtype_code, num_threads),
        data,
        mode,
        _AVIF_NATIVE_OUTPUT_MODES,
    )


def decode_heic(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode an HEIC/HEIF image into a ``[N]CHW`` tensor - requires ``libheif``!

    The output shape is ``(C, H, W)`` for a single-image HEIC and
    ``(N, C, H, W)`` for a multi-image one. All images must share the same
    dimensions and bit depth.

    .. important::

        HEIC decoding requires **libheif** to be installed and discoverable at
        runtime. TorchCodec does not bundle it (libheif is LGPL): install it via
        e.g. ``conda install -c conda-forge libheif``.

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_heic

            img = decode_heic("image.heic")

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded HEIC/HEIF data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. HEIC can store more than 8 bits per
            channel (e.g. 10- or 12-bit sources). ``torch.uint16`` always scales
            the samples up to fill the full 16-bit range ``[0, 65535]`` (8-bit
            0-255, 10-bit 0-1023 and 12-bit 0-4095 sources are all upscaled),
            while ``torch.uint8`` scales higher-bit sources down. ``"auto"``
            yields uint8 for 8-bit HEICs and uint16 (again filling ``[0, 65535]``)
            for higher-bit ones.

    Returns:
        torch.Tensor: The decoded image, of shape ``(C, H, W)`` (single-image)
        or ``(N, C, H, W)`` (multi-image).
    """
    output_dtype_code = _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    decode_heic_op = _get_decode_heic()
    decoded = _decode_to_mode(
        lambda d, m: decode_heic_op(d, m, output_dtype_code),
        data,
        mode,
        _HEIC_NATIVE_OUTPUT_MODES,
    )
    return _to_output_dtype(decoded, output_dtype)


def _detect_image_format(data: torch.Tensor) -> str:
    # Sniff the codec from the leading "magic" bytes of the encoded data.
    # This used to be implemented in C++ in torchvision, but benchmarks show
    # this is negligible in Python
    return _detect_image_format_from_header(bytes(data[:64].tolist()))


def _detect_image_format_from_header(header: bytes) -> str:
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
        # (animated) brands; HEIC/HEIF uses "heic"/"heix" (HEVC-coded) and the
        # generic "mif1"/"msf1" brands. We check avif first (it can also carry
        # mif1/msf1), then fall back to the HEIC-specific brands.
        brands = header[8:]
        if b"avif" in brands or b"avis" in brands:
            return "avif"
        if (
            b"heic" in brands
            or b"heix" in brands
            or b"heim" in brands
            or b"heis" in brands
            or b"hevc" in brands
            or b"hevx" in brands
            or b"mif1" in brands
            or b"msf1" in brands
        ):
            return "heic"
    raise ValueError(
        "Unsupported or unrecognized image format. Supported formats are "
        "JPEG, PNG, WebP, GIF, AVIF and HEIC. If you know you have a valid "
        "image, try using the dedicated decode_* functions like decode_jpeg() "
        "instead."
    )


# Design note: the parameters of decode_image must apply to *all* codecs
# uniformly. That's why all modes are supported by decode_image even though not
# all codec would natively support all mode - e.g. jpeg has no alpha support, so
# we prepend an opaque alpha channel as a post-processing step. As a resut, all
# codec-specific entry points like decode_jpeg, decode_png etc. must still
# expose the same parameters that decode_image exposes. The codec-specific
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
    """Decode an image into a ``[N]CHW`` tensor, detecting the format automatically.

    The format is detected from the encoded data (not the file extension), and
    decoding is delegated to the matching format-specific decoder. Supported
    formats are JPEG, PNG, WebP, GIF, AVIF and HEIC (requires ``libheif``). The
    output shape is ``(C, H, W)`` for a single image and ``(N, C, H, W)`` for
    animated or multi-image formats (WebP, GIF, AVIF, HEIC).

    For finer control, or for format-specific options (e.g. ``device`` for CUDA
    JPEG decoding, ``num_threads`` for AVIF), use the dedicated decoders
    directly: :func:`decode_jpeg`, :func:`decode_png`, :func:`decode_webp`,
    :func:`decode_gif`, :func:`decode_avif`, :func:`decode_heic`.

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_image

            jpeg_img = decode_image("image.jpg")
            png_img = decode_image("image.png")

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded image data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. Formats that can carry more than 8
            bits per channel (PNG, AVIF, HEIC) preserve that precision with
            ``torch.uint16`` and ``"auto"``. See the format-specific decoders
            for details.

    Returns:
        torch.Tensor: The decoded image, of shape ``[N]CHW``.
    """
    data = _source_to_tensor(source)
    format_to_decoder = {
        "jpeg": decode_jpeg,
        "png": decode_png,
        "webp": decode_webp,
        "gif": decode_gif,
        "avif": decode_avif,
        "heic": decode_heic,
    }
    fmt = _detect_image_format(data)
    return format_to_decoder[fmt](data, mode=mode, output_dtype=output_dtype)  # type: ignore[operator]


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata of an encoded image, as returned by :func:`get_image_metadata`.

    Every attribute describes the tensor that :func:`decode_image` (or the
    matching format-specific decoder) returns for this image.
    """

    format: str
    """The image format, e.g. ``"jpeg"``."""
    width: int
    """Width of the decoded image, **post-rotation**. This is the width of the
    tensor returned by :func:`decode_image` or the format-specific decoders,
    which always apply the EXIF orientation."""
    height: int
    """Height of the decoded image, **post-rotation**. This is the height of the
    tensor returned by :func:`decode_image` or the format-specific decoders,
    which always apply the EXIF orientation."""
    num_channels: int
    """Number of channels of the image. This is the number of channels
    of the decoded image when decoding with ``mode="UNCHANGED"``."""
    bit_depth: int
    """Number of bits per channel in the encoded image. When greater than 8,
    decoding with ``output_dtype="auto"`` returns a ``torch.uint16`` tensor."""
    num_frames: int
    """Number of frames (or images) in the file: 1 for still images, and the
    size of the leading ``N`` dimension of the decoded tensor for animated or
    multi-image files."""
    rotation: float | None
    """Rotation angle in degrees (counter-clockwise: one of 0, 90, -90 or 180)
    from the EXIF orientation metadata, or None if the image has no orientation
    metadata. The decoders always apply it, so the returned images are in the
    correct orientation.

    The EXIF orientation metadata can also specify a mirroring, which the decoders
    apply too but isn't reported here, since it doesn't change the image
    dimensions."""


class _ByteSource:
    # Random access over the encoded bytes. Paths are read lazily, so probing a
    # large file usually only reads its first few KB.

    def __init__(self, source):
        self._file = None
        if isinstance(source, (str, Path)):
            self._file = open(source, "rb")
            self._buf = None
            self.size = Path(source).stat().st_size
        elif isinstance(source, (bytes, bytearray)):
            self._buf = memoryview(source)
            self.size = len(source)
        elif isinstance(source, torch.Tensor):
            if (
                source.dtype != torch.uint8
                or source.ndim != 1
                or source.device.type != "cpu"
            ):
                raise ValueError(
                    "Input tensor must be a 1-dimensional uint8 CPU tensor, got "
                    f"dtype={source.dtype}, ndim={source.ndim}, device={source.device}."
                )
            # Zero-copy view of the tensor's data. The view doesn't own the
            # memory, so we keep a reference to the tensor. Slightly crazy, but
            # this is what gets us bytes (the python type) instead of
            # torch.uint8 which we'd have to copy to bytes.
            self._tensor = source.contiguous()
            self._buf = memoryview(
                (ctypes.c_ubyte * self._tensor.numel()).from_address(
                    self._tensor.data_ptr()
                )
            )
            self.size = len(self._buf)
        else:
            raise TypeError(
                f"Unknown source type: {type(source)}. "
                "Supported types are str, Path, bytes and torch.Tensor."
            )

    def read(self, offset: int, size: int) -> bytes:
        # Returns fewer than `size` bytes when reading past the end.
        if self._buf is not None:
            return bytes(self._buf[offset : offset + size])
        assert self._file is not None
        self._file.seek(offset)
        return self._file.read(size)

    def read_exact(self, offset: int, size: int) -> bytes:
        data = self.read(offset, size)
        if len(data) != size:
            raise RuntimeError("Image is truncated or corrupted.")
        return data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if self._file is not None:
            self._file.close()


_EXIF_INCORRECT_TAG = 0xFFFF


def _exif_orientation(tiff: bytes) -> int | None:
    # Port of fetch_exif_orientation() from Exif.h, quirks included, so that we
    # report exactly the orientation the decoders apply. `tiff` starts at the
    # TIFF header (i.e. after any "Exif\0\0" prefix). Returns the EXIF
    # orientation in [1, 8], or None if there's none (or it's invalid, in which
    # case the decoders don't apply anything either).
    n = len(tiff)
    byte_order = "<" if tiff[:2] == b"II" else ">"

    def u16(offset):
        if offset + 1 >= n:
            return _EXIF_INCORRECT_TAG
        return struct.unpack_from(byte_order + "H", tiff, offset)[0]

    if u16(2) != 0x2A:
        return None
    offset = (
        _EXIF_INCORRECT_TAG
        if 7 >= n
        else struct.unpack_from(byte_order + "I", tiff, 4)[0]
    )
    num_entries = u16(offset)
    offset += 2
    for _ in range(num_entries):
        tag = u16(offset)
        if tag == _EXIF_INCORRECT_TAG:
            break
        if tag == 0x0112:
            orientation = u16(offset + 8)
            return orientation if 1 <= orientation <= 8 else None
        offset += 12
    return None


# Rotation of each EXIF orientation, when decomposed into an optional
# left-to-right mirroring followed by a rotation.
_ROTATION_FROM_EXIF_ORIENTATION = {
    None: None,
    1: 0.0,
    2: 0.0,
    3: 180.0,
    4: 180.0,
    5: 90.0,
    6: -90.0,
    7: -90.0,
    8: 90.0,
}


# ----------------------------------------------------------------------------
# JPEG
# ----------------------------------------------------------------------------

# SOFn markers that libjpeg-turbo can decode. The other SOFn markers
# (hierarchical, and the reserved JPG marker) make jpeg_read_header() fail.
_JPEG_SUPPORTED_SOF = frozenset((0xC0, 0xC1, 0xC2, 0xC3, 0xC9, 0xCA, 0xCB))
_JPEG_UNSUPPORTED_SOF = frozenset((0xC5, 0xC6, 0xC7, 0xC8, 0xCD, 0xCE, 0xCF))


def _probe_jpeg(src: _ByteSource) -> ImageMetadata:
    # Walks the markers up to the first SOS, like jpeg_read_header(). The
    # decoder takes the EXIF orientation from the first APP1 segment, whatever
    # it contains.
    pos = 2  # skip SOI, already checked by the format sniffer
    sof = None
    orientation = None
    seen_app1 = False
    while True:
        chunk = src.read(pos, 4)
        if len(chunk) < 2:
            raise RuntimeError("JPEG is truncated: no SOS marker found.")
        if chunk[0] != 0xFF:
            # libjpeg skips garbage bytes between markers.
            window = src.read(pos, 4096)
            idx = window.find(b"\xff")
            pos += idx if idx > 0 else len(window)
            continue
        marker = chunk[1]
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker == 0xDA:  # SOS
            break
        if marker == 0xD9:  # EOI
            raise RuntimeError("JPEG has no image data.")
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # standalone markers
            pos += 2
            continue
        if len(chunk) < 4:
            raise RuntimeError("JPEG is truncated.")
        segment_length = (chunk[2] << 8) | chunk[3]
        if segment_length < 2:
            raise RuntimeError("Invalid JPEG marker segment length.")
        if marker in _JPEG_SUPPORTED_SOF:
            if sof is not None:
                raise RuntimeError("JPEG has more than one SOF marker.")
            precision, height, width, num_components = struct.unpack(
                ">BHHB", src.read_exact(pos + 4, 6)
            )
            if segment_length != 8 + 3 * num_components:
                raise RuntimeError("Invalid JPEG SOF marker length.")
            if height == 0 or width == 0 or num_components == 0:
                raise RuntimeError("Empty JPEG image (DNL not supported).")
            sof = (precision, height, width, num_components)
        elif marker in _JPEG_UNSUPPORTED_SOF:
            raise RuntimeError(f"Unsupported JPEG process: SOF type 0x{marker:02x}.")
        elif marker == 0xE1 and not seen_app1:
            seen_app1 = True
            payload = src.read_exact(pos + 4, segment_length - 2)
            if len(payload) > 6:
                orientation = _exif_orientation(payload[6:])
        pos += 2 + segment_length

    if sof is None:
        raise RuntimeError("JPEG has no SOF marker before SOS.")
    precision, height, width, num_components = sof
    if orientation is not None and 5 <= orientation <= 8:
        width, height = height, width
    return ImageMetadata(
        format="jpeg",
        width=width,
        height=height,
        num_channels=num_components,
        bit_depth=precision,
        num_frames=1,
        rotation=_ROTATION_FROM_EXIF_ORIENTATION[orientation],
    )


# ----------------------------------------------------------------------------
# PNG
# ----------------------------------------------------------------------------

_PNG_VALID_BIT_DEPTHS = {
    0: (1, 2, 4, 8, 16),  # gray
    2: (8, 16),  # RGB
    3: (1, 2, 4, 8),  # palette
    4: (8, 16),  # gray + alpha
    6: (8, 16),  # RGBA
}


def _probe_png(src: _ByteSource) -> ImageMetadata:
    # Reads IHDR and walks the chunks up to the first IDAT, which is what
    # png_read_info() sees. In particular, an eXIf chunk after IDAT is ignored
    # by the decoder.
    ihdr = src.read_exact(8, 8 + 13)
    length, chunk_type = struct.unpack_from(">I4s", ihdr)
    if chunk_type != b"IHDR" or length != 13:
        raise RuntimeError("PNG is missing its IHDR chunk.")
    width, height, bit_depth, color_type = struct.unpack_from(">IIBB", ihdr, 8)
    if not (0 < width < 2**31 and 0 < height < 2**31):
        raise RuntimeError(f"Invalid PNG dimensions: {width}x{height}.")
    if bit_depth not in _PNG_VALID_BIT_DEPTHS.get(color_type, ()):
        raise RuntimeError(
            f"Invalid PNG color type ({color_type}) and bit depth ({bit_depth})."
        )

    pos = 8 + 12 + 13
    num_palette = None
    has_trns = False
    exif = None
    while True:
        length, chunk_type = struct.unpack(">I4s", src.read_exact(pos, 8))
        if chunk_type == b"IDAT":
            break
        if chunk_type == b"IEND":
            raise RuntimeError("PNG has no IDAT chunk.")
        if chunk_type == b"PLTE":
            num_palette = length // 3
        elif chunk_type == b"tRNS" and not has_trns:
            # libpng ignores (with a warning) tRNS chunks that are invalid for
            # the color type.
            if color_type == 3:
                has_trns = num_palette is not None and 0 < length <= num_palette
            elif color_type == 0:
                has_trns = length == 2
            elif color_type == 2:
                has_trns = length == 6
        elif chunk_type == b"eXIf" and exif is None and length >= 2:
            payload = src.read_exact(pos + 8, length)
            if payload[0] == payload[1] and payload[0] in b"MI":
                exif = payload
        pos += 12 + length
    if color_type == 3 and num_palette is None:
        raise RuntimeError("PNG palette image has no PLTE chunk before IDAT.")

    orientation = _exif_orientation(exif) if exif is not None else None
    if orientation is not None and 5 <= orientation <= 8:
        width, height = height, width
    if color_type == 3:
        num_channels = 4 if has_trns else 3
        bit_depth = 8  # palette entries are 8-bit RGB(A)
    else:
        num_channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    return ImageMetadata(
        format="png",
        width=width,
        height=height,
        num_channels=num_channels,
        bit_depth=bit_depth,
        num_frames=1,
        rotation=_ROTATION_FROM_EXIF_ORIENTATION[orientation],
    )


# ----------------------------------------------------------------------------
# WebP
# ----------------------------------------------------------------------------

_WEBP_ANIMATION_FLAG = 0x02
_WEBP_EXIF_FLAG = 0x08
_WEBP_ALPHA_FLAG = 0x10


def _vp8_dimensions(src: _ByteSource, payload_pos: int) -> tuple[int, int]:
    header = src.read_exact(payload_pos, 10)
    if header[3:6] != b"\x9d\x01\x2a":
        raise RuntimeError("Invalid VP8 bitstream signature.")
    width, height = struct.unpack_from("<HH", header, 6)
    width, height = width & 0x3FFF, height & 0x3FFF
    if width == 0 or height == 0:
        raise RuntimeError("Invalid VP8 dimensions.")
    return width, height


def _vp8l_info(src: _ByteSource, payload_pos: int) -> tuple[int, int, bool]:
    header = src.read_exact(payload_pos, 5)
    if header[0] != 0x2F:
        raise RuntimeError("Invalid VP8L bitstream signature.")
    (bits,) = struct.unpack_from("<I", header, 1)
    if bits >> 29 != 0:
        raise RuntimeError("Invalid VP8L version.")
    width = (bits & 0x3FFF) + 1
    height = ((bits >> 14) & 0x3FFF) + 1
    has_alpha = bool((bits >> 28) & 1)
    return width, height, has_alpha


def _probe_webp(src: _ByteSource) -> ImageMetadata:
    # Mirrors WebPGetFeatures() (dimensions and alpha), WebPAnimDecoder (frame
    # count and canvas size) and WebPDemux (the EXIF chunk, which libwebpdemux
    # only exposes when the VP8X EXIF flag is set).
    first_type, first_size = struct.unpack("<4sI", src.read_exact(12, 8))
    orientation = None
    if first_type == b"VP8 ":
        width, height = _vp8_dimensions(src, 20)
        has_alpha = False
        num_frames = 1
    elif first_type == b"VP8L":
        width, height, has_alpha = _vp8l_info(src, 20)
        num_frames = 1
    elif first_type == b"VP8X":
        vp8x = src.read_exact(20, 10)
        flags = vp8x[0]
        canvas_width = int.from_bytes(vp8x[4:7], "little") + 1
        canvas_height = int.from_bytes(vp8x[7:10], "little") + 1
        is_animated = bool(flags & _WEBP_ANIMATION_FLAG)

        num_anmf = 0
        seen_alph = False
        bitstream = None  # (width, height, vp8l_alpha_or_None) of first image
        exif = None
        pos = 20 + first_size + (first_size & 1)
        while True:
            header = src.read(pos, 8)
            if len(header) < 8:
                break
            chunk_type, chunk_size = struct.unpack("<4sI", header)
            if chunk_type == b"ANMF":
                num_anmf += 1
            elif not is_animated and bitstream is None:
                if chunk_type == b"ALPH":
                    seen_alph = True
                elif chunk_type == b"VP8 ":
                    bitstream = (*_vp8_dimensions(src, pos + 8), None)
                elif chunk_type == b"VP8L":
                    bitstream = _vp8l_info(src, pos + 8)
            if chunk_type == b"EXIF" and exif is None and flags & _WEBP_EXIF_FLAG:
                exif = src.read_exact(pos + 8, chunk_size)
            pos += 8 + chunk_size + (chunk_size & 1)

        if exif is not None:
            if exif[:6] == b"Exif\x00\x00":
                exif = exif[6:]
            if len(exif) > 0:
                orientation = _exif_orientation(exif)

        if is_animated:
            if num_anmf == 0:
                raise RuntimeError("Animated WebP has no frames.")
            width, height = canvas_width, canvas_height
            has_alpha = bool(flags & _WEBP_ALPHA_FLAG)
            num_frames = num_anmf
        else:
            if bitstream is None:
                raise RuntimeError("WebP has no VP8 or VP8L bitstream.")
            width, height, vp8l_alpha = bitstream
            if (width, height) != (canvas_width, canvas_height):
                raise RuntimeError("WebP canvas and bitstream dimensions differ.")
            if vp8l_alpha is None:
                has_alpha = bool(flags & _WEBP_ALPHA_FLAG) or seen_alph
            else:
                has_alpha = vp8l_alpha or seen_alph
            num_frames = 1
    else:
        raise RuntimeError(f"Unexpected first WebP chunk: {first_type!r}.")

    if orientation is not None and 5 <= orientation <= 8:
        width, height = height, width
    return ImageMetadata(
        format="webp",
        width=width,
        height=height,
        num_channels=4 if has_alpha else 3,
        bit_depth=8,
        num_frames=num_frames,
        rotation=_ROTATION_FROM_EXIF_ORIENTATION[orientation],
    )


_PY_PROBES = {
    "jpeg": _probe_jpeg,
    "png": _probe_png,
    "webp": _probe_webp,
}


def get_image_metadata(source: str | Path | bytes | torch.Tensor) -> ImageMetadata:
    """Get the metadata of an encoded image, without decoding it.

    The format is detected from the encoded data, like in
    :func:`~torchcodec.decoders.decode_image`.

    See :class:`ImageMetadata` for the returned attributes, which all describe
    the output of the decoders.

    Example:

        .. code-block:: python

            from torchcodec.decoders import get_image_metadata

            metadata = get_image_metadata("image.jpg")
            print(metadata.width, metadata.height)

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded image data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.

    Returns:
        ImageMetadata: The image metadata.
    """
    with _ByteSource(source) as src:
        fmt = _detect_image_format_from_header(src.read(0, 64))
        if fmt not in _PY_PROBES:
            raise ValueError(
                f"get_image_metadata() doesn't support {fmt.upper()} images yet."
            )
        try:
            return _PY_PROBES[fmt](src)
        except (struct.error, IndexError):
            raise RuntimeError("Image is truncated or corrupted.") from None
