# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for using TorchCodec when FFmpeg is NOT installed.

FFmpeg is an optional runtime dependency: ``import torchcodec``, the image
decoders and (eventually) other FFmpeg-free features must work without it, while
the FFmpeg-backed entry points (VideoDecoder, AudioDecoder, WavDecoder, the
encoders) must fail with a clear error only when they are used.

The whole module self-skips when FFmpeg *is* available, so it is a no-op in the
regular test jobs and only does something meaningful in the dedicated
no-FFmpeg CI job (which installs no FFmpeg at all).
"""

import pytest
import torch

import torchcodec
from torchcodec._core import ops as _ops
from torchcodec.decoders import AudioDecoder, VideoDecoder, WavDecoder
from torchcodec.encoders import AudioEncoder, Encoder, VideoEncoder

from .utils import (
    avif_is_available,
    GRADIENT_AVIF,
    GRADIENT_GIF,
    GRADIENT_JPEG,
    GRADIENT_PNG,
    GRADIENT_WEBP,
    jpeg_is_available,
    png_is_available,
    webp_is_available,
)

pytestmark = pytest.mark.skipif(
    _ops._FFMPEG_AVAILABLE,
    reason="These tests validate TorchCodec's behavior when FFmpeg is NOT available.",
)


def test_ffmpeg_is_really_absent():
    # Sanity check that we are genuinely in the no-FFmpeg configuration.
    assert _ops._FFMPEG_AVAILABLE is False
    assert torchcodec.ffmpeg_major_version is None
    assert torchcodec.core_library_path is None


@pytest.mark.parametrize(
    "decode_fn_name, asset, is_available",
    [
        ("decode_jpeg", GRADIENT_JPEG, jpeg_is_available),
        ("decode_png", GRADIENT_PNG, png_is_available),
        ("decode_webp", GRADIENT_WEBP, webp_is_available),
        ("decode_gif", GRADIENT_GIF, lambda: True),
        ("decode_avif", GRADIENT_AVIF, avif_is_available),
    ],
)
def test_image_decoders_work_without_ffmpeg(decode_fn_name, asset, is_available):
    if not is_available():
        pytest.skip(f"{decode_fn_name} codec support isn't available.")

    from torchcodec.decoders import _image_decoders

    decode_fn = getattr(_image_decoders, decode_fn_name)
    out = decode_fn(asset.path)
    # Image decoders return (C, H, W) for still images.
    assert out.shape[-2:] == (asset.height, asset.width)
    assert out.dtype == torch.uint8


def test_decode_image_works_without_ffmpeg():
    if not jpeg_is_available():
        pytest.skip("libjpeg support isn't available.")

    from torchcodec.decoders._image_decoders import decode_image

    out = decode_image(GRADIENT_JPEG.path)
    assert out.shape == (
        GRADIENT_JPEG.num_channels,
        GRADIENT_JPEG.height,
        GRADIENT_JPEG.width,
    )


# The entry-point classes are imported at module scope (above): those imports
# must succeed even without FFmpeg. Each entry below only *constructs* an object,
# so the pytest.raises block unambiguously asserts that construction is what
# raises (not the import).
@pytest.mark.parametrize(
    "make_entry_point",
    [
        pytest.param(lambda: VideoDecoder(GRADIENT_JPEG.path), id="VideoDecoder"),
        pytest.param(lambda: AudioDecoder(GRADIENT_JPEG.path), id="AudioDecoder"),
        pytest.param(lambda: WavDecoder(GRADIENT_JPEG.path), id="WavDecoder"),
        pytest.param(
            lambda: VideoEncoder(
                torch.zeros(1, 3, 16, 16, dtype=torch.uint8), frame_rate=30
            ),
            id="VideoEncoder",
        ),
        pytest.param(
            lambda: AudioEncoder(torch.zeros(1, 100), sample_rate=16000),
            id="AudioEncoder",
        ),
        pytest.param(lambda: Encoder(), id="Encoder"),
    ],
)
def test_ffmpeg_entry_points_raise_clear_error(make_entry_point):
    # Every FFmpeg-backed entry point must raise a clear, actionable error
    # explaining that FFmpeg could not be loaded, and it must do so at
    # construction time (not silently succeed and fail later).
    with pytest.raises(RuntimeError, match="Could not load libtorchcodec"):
        make_entry_point()
