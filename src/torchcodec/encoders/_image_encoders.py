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
    def __init__(self, img: Tensor) -> None:
        self._img = img

    def to_file(self, dest: str | Path, *, quality: int = 75) -> None:
        self._validate_quality(quality)
        _encode_jpeg_to_file(self._img, str(dest), quality)

    def to_file_like(
        self, dest: io.RawIOBase | io.BufferedIOBase, *, quality: int = 75
    ) -> None:
        self._validate_quality(quality)
        _encode_jpeg_to_file_like(
            self._img, create_file_like_context(dest, True), quality
        )

    def to_tensor(self, *, quality: int = 75) -> Tensor:
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
    def __init__(self, img: Tensor) -> None:
        self._img = img

    def to_file(self, dest: str | Path, *, compression_level: int = 6) -> None:
        _encode_png_to_file(self._img, str(dest), compression_level)

    def to_file_like(
        self, dest: io.RawIOBase | io.BufferedIOBase, *, compression_level: int = 6
    ) -> None:
        _encode_png_to_file_like(
            self._img, create_file_like_context(dest, True), compression_level
        )

    def to_tensor(self, *, compression_level: int = 6) -> Tensor:
        return _encode_to_tensor_through_bytesio(
            self._img, compression_level, _encode_png_to_file_like
        )
