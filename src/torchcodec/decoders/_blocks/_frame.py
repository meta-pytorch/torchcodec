# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from torchcodec._core.ops import _blocks_frame_to_planes


class Packet:
    """Opaque, thread-movable handle to a demuxed (compressed) packet.

    Produced by :class:`Demuxer`, consumed by :class:`PacketDecoder`. It wraps a raw
    pointer, so it is only valid within the process that created it (it cannot
    cross a process boundary).
    """

    def __init__(self, handle: torch.Tensor):
        self._handle = handle


@dataclass
class RawFrame:
    """A decoded frame's own samples, as zero-copy views, before any color
    conversion.

    ``planes`` holds one strided view per component, in the order given by
    ``component_names``. Nothing is copied, so the samples are exactly what the
    codec produced -- including the full precision of 10- and 12-bit (HDR)
    content -- and the views live on the device the frame was decoded on. They
    keep the frame alive, so they stay valid after the :class:`DecodedFrame` they
    came from is dropped.

    Not every video is YUV. ``component_names`` is ``("Y", "U", "V")`` for the
    common case, but it may be ``("Y",)`` for grayscale, ``("R", "G", "B")`` for
    RGB codecs, or carry a trailing ``"A"`` for formats with alpha; check
    ``is_rgb`` rather than assuming. A handful of layouts can't be exposed
    without a copy (palettised, sub-byte-packed, or bottom-up frames) and raise
    instead.

    The views are often non-contiguous, because that is what avoids the copy:
    for 4:2:0 content the chroma planes are half-resolution, and for packed or
    semi-planar formats (NV12 and P010/P016 -- what NVDEC produces -- but also
    yuyv422, bgra...) the components are interleaved in memory and come back
    with a sample stride greater than 1. Call ``.contiguous()`` if you need
    packed planes; that is the only point at which a copy happens.

    ``bit_depth`` is the number of significant bits per sample. When it is
    smaller than the number of bits the pixel format stores, ``msb_aligned``
    tells you where they sit: NVDEC decodes 10-bit content into 16-bit P016
    surfaces with the samples in the *high* bits, so a sample's value is
    ``raw >> (container_bit_depth - bit_depth)``. Software decoding of the same
    content yields ``yuv420p10le``, which needs no shift.
    """

    planes: tuple[torch.Tensor, ...]
    component_names: tuple[str, ...]
    pix_fmt: str
    is_rgb: bool
    bit_depth: int
    container_bit_depth: int
    colorspace: int
    color_range: int
    pts_seconds: float
    duration_seconds: float

    @property
    def msb_aligned(self) -> bool:
        return self.container_bit_depth > self.bit_depth

    def __getitem__(self, name: str) -> torch.Tensor:
        """The plane of a named component, e.g. ``raw["Y"]``."""
        try:
            return self.planes[self.component_names.index(name)]
        except ValueError:
            raise KeyError(
                f"{self.pix_fmt} has no {name} component, only "
                f"{', '.join(self.component_names)}."
            ) from None


class DecodedFrame:
    """A decoded (YUV) frame: an opaque, thread-movable handle to the raw frame
    plus its presentation timestamp and duration (in seconds).

    Produced by :class:`PacketDecoder`, consumed by :class:`ColorConverter`. The
    handle wraps a raw pointer and is process-local. pts/duration are stamped by
    the decoder (which knows the stream time base) and carried here so the
    :class:`ColorConverter` need not be bound to any stream.

    Use :meth:`to_planes` to get at the raw samples instead of color-converting.

    ``device`` is where this frame's samples actually are, which is not
    necessarily the device its decoder was created with: a CUDA decoder falls
    back to CPU decoding for streams NVDEC can't handle.
    """

    def __init__(
        self,
        handle: torch.Tensor,
        pts_seconds: float,
        duration_seconds: float,
        device: str = "cpu",
        bit_depth: int = 8,
    ):
        self._handle = handle
        self._device = device
        self._bit_depth = bit_depth
        self.pts_seconds = pts_seconds
        self.duration_seconds = duration_seconds

    @property
    def device(self) -> str:
        return self._device

    def to_planes(self) -> RawFrame:
        """Return this frame's own samples as a :class:`RawFrame`, without
        copying and without color-converting."""
        *planes, metadata = _blocks_frame_to_planes(self._handle, self._device)
        parsed = json.loads(metadata)
        names = tuple(parsed["components"])
        return RawFrame(
            planes=tuple(planes[: len(names)]),
            component_names=names,
            pix_fmt=parsed["pix_fmt"],
            is_rgb=parsed["is_rgb"],
            bit_depth=self._bit_depth,
            container_bit_depth=parsed["container_bit_depth"],
            colorspace=parsed["colorspace"],
            color_range=parsed["color_range"],
            pts_seconds=self.pts_seconds,
            duration_seconds=self.duration_seconds,
        )
