# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from torchcodec._core.ops import _blocks_frame_metadata, _blocks_frame_planes


class _Metadata(NamedTuple):
    # The fields of the `_blocks_frame_metadata` op, in order.
    pix_fmt: str
    colorspace: str
    color_range: str
    bit_depth: int
    width: int
    height: int
    rotation_degrees: float


class Packet:
    def __init__(
        self,
        handle: torch.Tensor,
        stream_index: int,
        *,
        generation: int = 0,
    ):
        self._handle = handle
        self.stream_index = stream_index
        # Which side of the demuxer's last seek this packet came from. Private:
        # it exists so a decoder can catch a missing reset(), not for callers to
        # reason about. See _BasePacketDecoder.decode().
        self._generation = generation


class RawFrame:
    def __init__(
        self,
        handle: torch.Tensor,
        pts_seconds: float,
        duration_seconds: float,
        storage: torch.Tensor | None = None,
    ):
        self._handle = handle
        self._storage = storage
        self.pts_seconds = pts_seconds
        self.duration_seconds = duration_seconds
        self._metadata: _Metadata | None = None
        self._planes: tuple[torch.Tensor, ...] | None = None

    @property
    def _device(self) -> torch.device:
        return (
            self._storage.device if self._storage is not None else torch.device("cpu")
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        # See [Standalone Frame Storage and the need for record_stream]
        if self._storage is not None:
            self._storage.record_stream(stream)

    def _get_metadata(self) -> _Metadata:
        if self._metadata is None:
            self._metadata = _Metadata(*_blocks_frame_metadata(self._handle))
        return self._metadata

    @property
    def pix_fmt(self) -> str:
        return self._get_metadata().pix_fmt

    @property
    def colorspace(self) -> str:
        return self._get_metadata().colorspace

    @property
    def color_range(self) -> str:
        return self._get_metadata().color_range

    @property
    def bit_depth(self) -> int:
        return self._get_metadata().bit_depth

    @property
    def width(self) -> int:
        return self._get_metadata().width

    @property
    def height(self) -> int:
        return self._get_metadata().height

    @property
    def rotation_degrees(self) -> float:
        return self._get_metadata().rotation_degrees

    @property
    def planes(self) -> tuple[torch.Tensor, ...]:
        if self._planes is None:
            planes = _blocks_frame_planes(self._handle, self._device)
            # The op's schema is fixed-arity, so it pads with empty tensors up
            # to the 4 components of the widest format.
            self._planes = tuple(p for p in planes if p.numel() > 0)
        return self._planes


@dataclass
class RawAudioSamples:
    data: torch.Tensor
    sample_rate: int
    pts_seconds: float
    duration_seconds: float
    # See Packet._generation.
    _generation: int = 0

    @property
    def num_channels(self) -> int:
        return self.data.shape[0]

    @property
    def num_samples(self) -> int:
        return self.data.shape[1]
