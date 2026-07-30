# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch

from torchcodec._core.ops import (
    create_file_like_context,
    encode_jpeg_to_file as _encode_jpeg_to_file,
    encode_jpeg_to_file_like as _encode_jpeg_to_file_like,
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
    dest: str | Path,
    quality: int = 75,
) -> None:
    """Encode a CHW uint8 image tensor into a JPEG file.

    Args:
        input (``torch.Tensor``): The image to encode, a 3-dimensional uint8
            tensor in CHW layout with 1 (grayscale) or 3 (RGB) channels.
        dest (str, ``pathlib.Path`` or file-like object): The destination to
            write the encoded JPEG to. Either a path to the output file, or a
            file-like object that supports ``write(data: bytes) -> int`` and
            ``seek(offset: int, whence: int = 0) -> int``, such as
            ``io.BytesIO()`` or an open file in binary write mode.
        quality (int): Quality of the resulting JPEG, between 1 and 100. Higher
            means better quality and larger file size. Default: 75.

    If ``input`` is on a CUDA device, encoding is performed on the GPU with
    nvJPEG. Only 3-channel RGB tensors are supported on CUDA (grayscale must be
    encoded on the CPU).
    """
    if quality < 1 or quality > 100:
        raise ValueError("Image quality should be a positive number between 1 and 100")
    _encode_to_dest(
        input,
        dest,
        quality,
        to_file=_encode_jpeg_to_file,
        to_file_like=_encode_jpeg_to_file_like,
    )
