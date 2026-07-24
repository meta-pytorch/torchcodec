# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for using TorchCodec when FFmpeg is NOT installed.

FFmpeg is an optional runtime dependency: ``import torchcodec``, the image
decoders, while the FFmpeg-backed entry points must fail with a clear error only
when they are used.

The whole module self-skips when FFmpeg is available, so it is a no-op in the
regular test jobs and only does something meaningful in the dedicated no-FFmpeg
CI job (which installs no FFmpeg, and verifies that it's not available.).
"""

import pytest
import torch

import torchcodec
from torchcodec._core import ops as _ops
from torchcodec.decoders import AudioDecoder, VideoDecoder, WavDecoder
from torchcodec.decoders._image_decoders import (
    decode_avif,
    decode_gif,
    decode_image,
    decode_jpeg,
    decode_png,
    decode_webp,
)
from torchcodec.encoders import AudioEncoder, Encoder, VideoEncoder

from .utils import (
    GRADIENT_AVIF,
    GRADIENT_GIF,
    GRADIENT_JPEG,
    GRADIENT_PNG,
    GRADIENT_WEBP,
)

pytestmark = pytest.mark.skipif(
    _ops._FFMPEG_AVAILABLE,
    reason="These tests validate TorchCodec's behavior when FFmpeg is NOT available.",
)


def test_ffmpeg_is_really_absent():
    assert _ops._FFMPEG_AVAILABLE is False
    assert torchcodec.ffmpeg_major_version is None
    assert torchcodec.core_library_path is None


@pytest.mark.parametrize(
    "decode_fn, asset",
    [
        pytest.param(
            decode_jpeg, GRADIENT_JPEG, marks=pytest.mark.needs_jpeg, id="jpeg"
        ),
        pytest.param(decode_png, GRADIENT_PNG, marks=pytest.mark.needs_png, id="png"),
        pytest.param(
            decode_webp, GRADIENT_WEBP, marks=pytest.mark.needs_webp, id="webp"
        ),
        pytest.param(decode_gif, GRADIENT_GIF, id="gif"),
        pytest.param(
            decode_avif, GRADIENT_AVIF, marks=pytest.mark.needs_avif, id="avif"
        ),
        pytest.param(
            decode_image, GRADIENT_JPEG, marks=pytest.mark.needs_jpeg, id="image"
        ),
    ],
)
def test_image_decoders_work_without_ffmpeg(decode_fn, asset):
    out = decode_fn(asset.path)
    assert out.shape[-2:] == (asset.height, asset.width)
    assert out.dtype == torch.uint8


@pytest.mark.parametrize(
    "make_entry_point",
    [
        # we can pass "irrelevant" source because construction fails before it's seen
        pytest.param(lambda: VideoDecoder("irrelevant"), id="VideoDecoder"),
        pytest.param(lambda: AudioDecoder("irrelevant"), id="AudioDecoder"),
        pytest.param(lambda: WavDecoder("irrelevant"), id="WavDecoder"),
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
    with pytest.raises(RuntimeError, match="Could not load libtorchcodec"):
        make_entry_point()
