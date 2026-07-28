# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchcodec._core.ops import encode_jpeg as _encode_jpeg, encode_png as _encode_png


def encode_png(input: torch.Tensor, compression_level: int = 6) -> torch.Tensor:
    """Encode a CHW uint8 image tensor into the bytes of a PNG file.

    Args:
        input (``torch.Tensor``): The image to encode, a 3-dimensional uint8
            tensor in CHW layout with 1 (grayscale) or 3 (RGB) channels.
        compression_level (int): zlib compression level between 0 (no
            compression, fastest) and 9 (max compression, slowest). Default: 6.

    Returns:
        torch.Tensor: A 1-dimensional uint8 tensor holding the raw bytes of the
            encoded PNG file.
    """
    return _encode_png(input, compression_level)


def encode_jpeg(input: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Encode a CHW uint8 image tensor into the bytes of a JPEG file.

    Args:
        input (``torch.Tensor``): The image to encode, a 3-dimensional uint8
            tensor in CHW layout with 1 (grayscale) or 3 (RGB) channels.
        quality (int): Quality of the resulting JPEG, between 1 and 100. Higher
            means better quality and larger file size. Default: 75.

    Returns:
        torch.Tensor: A 1-dimensional uint8 tensor holding the raw bytes of the
            encoded JPEG file.
    """
    if quality < 1 or quality > 100:
        raise ValueError("Image quality should be a positive number between 1 and 100")
    return _encode_jpeg(input, quality)
