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


def _encode_to_tensor_through_bytesio(input, param, to_file_like) -> torch.Tensor:
    # Encode into an in-memory BytesIO and wrap its buffer as a 1-D uint8 tensor.
    # getbuffer() (unlike getvalue()) exposes the buffer without copying. We
    # could have native C++ implementation for that in each encoder, but it's
    # not always worth it (based on benchmarks). Currently, the only encoder
    # that really needs a dedicated C++ path is JPEG on CUDA.
    buf = io.BytesIO()
    to_file_like(input, create_file_like_context(buf, True), param)
    return torch.frombuffer(buf.getbuffer(), dtype=torch.uint8)


def encode_png(
    input: torch.Tensor,
    dest: str | Path | None = None,
    compression_level: int = 6,
) -> torch.Tensor | None:
    """Encode a CHW uint8 image tensor into a PNG.

    Args:
        input (``torch.Tensor``): The image to encode, a 3-dimensional uint8
            tensor in CHW layout with 1 (grayscale) or 3 (RGB) channels.
        dest (str, ``pathlib.Path``, file-like object, or ``None``): The
            destination to write the encoded PNG to. Either a path to the output
            file, or a file-like object that supports
            ``write(data: bytes) -> int`` and
            ``seek(offset: int, whence: int = 0) -> int``, such as
            ``io.BytesIO()`` or an open file in binary write mode. If ``None``
            (the default), the encoded bytes are returned as a 1-D uint8 tensor
            instead of being written anywhere.
        compression_level (int): zlib compression level between 0 (no
            compression, fastest) and 9 (max compression, slowest). Default: 6.

    Returns:
        ``None`` if ``dest`` is a path or file-like object, otherwise a 1-D uint8
        tensor of the encoded bytes.
    """
    if dest is None:
        return _encode_to_tensor_through_bytesio(
            input, compression_level, _encode_png_to_file_like
        )
    else:
        _encode_to_dest(
            input,
            dest,
            compression_level,
            to_file=_encode_png_to_file,
            to_file_like=_encode_png_to_file_like,
        )
        return None


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
        if input.is_cuda:
            return _encode_jpeg_to_tensor_cuda(input, quality)
        else:
            return _encode_to_tensor_through_bytesio(
                input, quality, _encode_jpeg_to_file_like
            )
    else:
        _encode_to_dest(
            input,
            dest,
            quality,
            to_file=_encode_jpeg_to_file,
            to_file_like=_encode_jpeg_to_file_like,
        )
        return None
