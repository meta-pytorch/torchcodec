# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
from pathlib import Path

import torch
from torch import Tensor

from torchcodec._core.ops import (
    create_file_like_context,
    encode_jpeg_to_file as _encode_jpeg_to_file,
    encode_jpeg_to_file_like as _encode_jpeg_to_file_like,
    encode_jpeg_to_tensor_cuda as _encode_jpeg_to_tensor_cuda,
    encode_png_to_file as _encode_png_to_file,
    encode_png_to_file_like as _encode_png_to_file_like,
)


def _encode_to_tensor_through_bytesio(img, param, to_file_like) -> Tensor:
    # Encode into an in-memory BytesIO and wrap its buffer as a 1-D uint8 tensor.
    # getbuffer() (unlike getvalue()) exposes the buffer without copying. We
    # could have native C++ implementation for that in each encoder, but it's
    # not always worth it (based on benchmarks). Currently, the only encoder
    # that really needs a dedicated C++ path is JPEG on CUDA.
    buf = io.BytesIO()
    to_file_like(img, create_file_like_context(buf, True), param)
    return torch.frombuffer(buf.getbuffer(), dtype=torch.uint8)


class JpegEncoder:
    """Encoder for JPEG images.

    Example:

        .. code-block:: python

            from torchcodec.encoders import JpegEncoder

            JpegEncoder(img).to_file("image.jpg")
            # or encode to a file-like object or to a tensor, see methods below.

    Args:
        img (``torch.Tensor``): The image to encode, a 3-dimensional uint8 tensor
            in CHW layout with 1 (grayscale) or 3 (RGB) channels. If on a CUDA
            device, encoding is performed on the GPU with nvJPEG, and only
            3-channel RGB is supported.
    """

    def __init__(self, img: Tensor) -> None:
        self._img = img

    def to_file(self, dest: str | Path, *, quality: int = 75) -> None:
        """Encode the image into a JPEG file.

        Args:
            dest (str or ``pathlib.Path``): The path to the output file, e.g.
                ``image.jpg``.
            quality (int, optional): Quality of the output, between 1 and 100.
                Higher means better quality and larger file size. Default: 75.
        """
        self._validate_quality(quality)
        _encode_jpeg_to_file(self._img, str(dest), quality)

    def to_file_like(
        self, dest: io.RawIOBase | io.BufferedIOBase, *, quality: int = 75
    ) -> None:
        """Encode the image into a file-like object.

        Args:
            dest: A writable file-like object supporting ``write`` and ``seek``,
                such as ``io.BytesIO()`` or an open file in binary write mode.
            quality (int, optional): Quality of the output, between 1 and 100.
                Higher means better quality and larger file size. Default: 75.
        """
        self._validate_quality(quality)
        _encode_jpeg_to_file_like(
            self._img, create_file_like_context(dest, True), quality
        )

    def to_tensor(self, *, quality: int = 75) -> Tensor:
        """Encode the image into raw bytes, as a 1D uint8 tensor.

        The returned tensor is on the same device as the input image (a CUDA
        input yields a CUDA tensor).

        Args:
            quality (int, optional): Quality of the output, between 1 and 100.
                Higher means better quality and larger file size. Default: 75.

        Returns:
            torch.Tensor: The encoded bytes, a 1D uint8 tensor on the same
            device as the input image.
        """
        self._validate_quality(quality)
        if self._img.is_cuda:
            return _encode_jpeg_to_tensor_cuda(self._img, quality)
        else:
            return _encode_to_tensor_through_bytesio(
                self._img, quality, _encode_jpeg_to_file_like
            )

    @staticmethod
    def _validate_quality(quality: int) -> None:
        if quality < 1 or quality > 100:
            raise ValueError(
                "Image quality should be a positive number between 1 and 100"
            )


class PngEncoder:
    """Encoder for PNG images.

    Example:

        .. code-block:: python

            from torchcodec.encoders import PngEncoder

            PngEncoder(img).to_file("image.png")
            # or encode to a file-like object or to a tensor, see methods below.

    Args:
        img (``torch.Tensor``): The image to encode, a 3-dimensional uint8 tensor
            in CHW layout with 1 (grayscale) or 3 (RGB) channels.
    """

    def __init__(self, img: Tensor) -> None:
        self._img = img

    def to_file(self, dest: str | Path, *, compression_level: int = 6) -> None:
        """Encode the image into a PNG file.

        Args:
            dest (str or ``pathlib.Path``): The path to the output file, e.g.
                ``image.png``.
            compression_level (int, optional): zlib compression level between 0
                (no compression, fastest) and 9 (max compression, slowest).
                Default: 6.
        """
        _encode_png_to_file(self._img, str(dest), compression_level)

    def to_file_like(
        self, dest: io.RawIOBase | io.BufferedIOBase, *, compression_level: int = 6
    ) -> None:
        """Encode the image into a file-like object.

        Args:
            dest: A writable file-like object supporting ``write`` and ``seek``,
                such as ``io.BytesIO()`` or an open file in binary write mode.
            compression_level (int, optional): zlib compression level between 0
                (no compression, fastest) and 9 (max compression, slowest).
                Default: 6.
        """
        _encode_png_to_file_like(
            self._img, create_file_like_context(dest, True), compression_level
        )

    def to_tensor(self, *, compression_level: int = 6) -> Tensor:
        """Encode the image into raw bytes, as a 1D uint8 tensor.

        Args:
            compression_level (int, optional): zlib compression level between 0
                (no compression, fastest) and 9 (max compression, slowest).
                Default: 6.

        Returns:
            torch.Tensor: The encoded bytes, a 1D uint8 tensor.
        """
        return _encode_to_tensor_through_bytesio(
            self._img, compression_level, _encode_png_to_file_like
        )
