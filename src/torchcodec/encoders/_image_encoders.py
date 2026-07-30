# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
from pathlib import Path

import torch

from torchcodec._core.ops import (
    create_file_like_context,
    encode_jpeg_to_file as _encode_jpeg_to_file,
    encode_jpeg_to_file_like as _encode_jpeg_to_file_like,
    encode_jpeg_to_tensor_cuda as _encode_jpeg_to_tensor_cuda,
    encode_png_to_file as _encode_png_to_file,
    encode_png_to_file_like as _encode_png_to_file_like,
)


def _encode_to_dest(input, dest, param, *, to_file, to_file_like) -> None:
    if isinstance(dest, (str, Path)):
        to_file(input, str(dest), param)
    else:
        # Assume file-like, it gets validated in C++ (it's tested).
        to_file_like(input, create_file_like_context(dest, True), param)


def encode_png(
    input: torch.Tensor,
    dest: str | Path,
    compression_level: int = 6,
) -> None:
    """Encode a CHW uint8 image tensor into a PNG file.

    Args:
        input (``torch.Tensor``): The image to encode, a 3-dimensional uint8
            tensor in CHW layout with 1 (grayscale) or 3 (RGB) channels.
        dest (str, ``pathlib.Path`` or file-like object): The destination to
            write the encoded PNG to. Either a path to the output file, or a
            file-like object that supports ``write(data: bytes) -> int`` and
            ``seek(offset: int, whence: int = 0) -> int``, such as
            ``io.BytesIO()`` or an open file in binary write mode.
        compression_level (int): zlib compression level between 0 (no
            compression, fastest) and 9 (max compression, slowest). Default: 6.
    """
    _encode_to_dest(
        input,
        dest,
        compression_level,
        to_file=_encode_png_to_file,
        to_file_like=_encode_png_to_file_like,
    )


def encode_jpeg(
    input: torch.Tensor,
    dest: str | Path | None = None,
    quality: int = 75,
) -> torch.Tensor | None:
    """Encode a CHW uint8 image tensor into a JPEG.

    Args:
        input (``torch.Tensor``): The image to encode, a 3-dimensional uint8
            tensor in CHW layout with 1 (grayscale) or 3 (RGB) channels.
        dest (str, ``pathlib.Path``, file-like object, or ``None``): The
            destination to write the encoded JPEG to. Either a path to the output
            file, or a file-like object that supports
            ``write(data: bytes) -> int`` and
            ``seek(offset: int, whence: int = 0) -> int``, such as
            ``io.BytesIO()`` or an open file in binary write mode. If ``None``
            (the default), the encoded bytes are returned as a 1-D uint8 tensor
            instead of being written anywhere.
        quality (int): Quality of the resulting JPEG, between 1 and 100. Higher
            means better quality and larger file size. Default: 75.

    Returns:
        ``None`` if ``dest`` is a path or file-like object. If ``dest`` is
        ``None``, a 1-D uint8 tensor of the encoded bytes, on the same device as
        ``input`` (a CUDA input yields a CUDA tensor; call ``.cpu()`` for host
        bytes).

    If ``input`` is on a CUDA device, encoding is performed on the GPU with
    nvJPEG. Only 3-channel RGB tensors are supported on CUDA (grayscale must be
    encoded on the CPU).
    """
    if quality < 1 or quality > 100:
        raise ValueError("Image quality should be a positive number between 1 and 100")

    if dest is None:
        # TODO_IMAGE since we're going to expose to-tensor support, we should
        # probably see if we can benefit for custom implementations like this
        # one with nvjpeg, but for the other 'backends' (png CPU and jpeg CPU)
        # vs the currently-recommended way to go through BytesIO + getbuffer().
        if input.is_cuda:
            # Zero-copy: the encoded bytes are retrieved straight into a CUDA
            # tensor and never leave the GPU.
            return _encode_jpeg_to_tensor_cuda(input, quality)
        # On CPU, encode into a BytesIO and wrap its buffer without an extra copy
        # (getbuffer() doesn't copy, unlike getvalue()).
        buf = io.BytesIO()
        _encode_jpeg_to_file_like(input, create_file_like_context(buf, True), quality)
        return torch.frombuffer(buf.getbuffer(), dtype=torch.uint8)

    _encode_to_dest(
        input,
        dest,
        quality,
        to_file=_encode_jpeg_to_file,
        to_file_like=_encode_jpeg_to_file_like,
    )
    return None
