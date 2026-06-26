# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

import torchcodec
from torchcodec.decoders import VideoDecoder

from .utils import NASA_VIDEO


@pytest.fixture(autouse=True)
def _restore_bridge():
    # Keep tests isolated: restore the default bridge afterwards.
    previous = torchcodec.get_bridge()
    yield
    torchcodec.set_bridge(previous)


def test_default_bridge_is_torch_when_torch_installed():
    assert torchcodec.get_bridge() == "torch"
    decoder = VideoDecoder(NASA_VIDEO.path)
    assert isinstance(decoder[0], torch.Tensor)
    assert isinstance(decoder.get_frames_in_range(0, 4).data, torch.Tensor)


def test_numpy_bridge_returns_numpy():
    torchcodec.set_bridge("numpy")
    decoder = VideoDecoder(NASA_VIDEO.path)

    # Single frame.
    assert isinstance(decoder[0], np.ndarray)
    assert decoder[0].dtype == np.uint8

    frame = decoder.get_frame_at(5)
    assert isinstance(frame.data, np.ndarray)
    assert isinstance(frame.pts_seconds, float)

    # Batches: data + pts + duration are all numpy.
    for batch in (
        decoder.get_frames_in_range(0, 6, 2),
        decoder.get_frames_at([0, 3, 5]),
        decoder.get_frames_played_at([0.0, 0.1]),
        decoder.get_frames_played_in_range(0.0, 0.3),
    ):
        assert isinstance(batch.data, np.ndarray)
        assert isinstance(batch.pts_seconds, np.ndarray)
        assert isinstance(batch.duration_seconds, np.ndarray)


def test_numpy_bridge_matches_torch_values():
    decoder = VideoDecoder(NASA_VIDEO.path)
    torch_frame = decoder[10]
    torchcodec.set_bridge("numpy")
    numpy_frame = VideoDecoder(NASA_VIDEO.path)[10]
    np.testing.assert_array_equal(torch_frame.numpy(), numpy_frame)


def test_set_bridge_validation():
    with pytest.raises(ValueError, match="Invalid bridge"):
        torchcodec.set_bridge("tensorflow")
