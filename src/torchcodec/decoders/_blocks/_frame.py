# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

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
    conversion. Returned by :meth:`DecodedFrame.materialize`.

    ``planes`` holds one strided view per component, in the frame's native
    order: ``(Y, U, V)`` for the common YUV formats, ``(R, G, B)`` for RGB
    codecs, ``(Y,)`` for grayscale, plus a trailing alpha view when the format
    has one. Chroma planes are subsampled (half-resolution for 4:2:0) and the
    views are usually non-contiguous, which is what avoids the copy.

    Samples of more than 8 bits come back as uint16, with ``bit_depth``
    significant bits. Where those bits sit within the 16 is a property of
    ``pix_fmt``: msb-aligned for p010le/p012le, lsb-aligned for yuv420p10le.
    """

    planes: tuple[torch.Tensor, ...]
    pix_fmt: str  # FFmpeg pixel-format name, e.g. "yuv420p"
    colorspace: str  # e.g. "bt709"
    color_range: str  # "tv" (limited) or "pc" (full)
    bit_depth: int


# TODO_API_BREAKDOWN P1: API design - especially the materialize() method but
# also the public fields, class name etc.
class DecodedFrame:
    """A decoded (YUV) frame: an opaque, thread-movable handle to the raw frame
    plus its presentation timestamp and duration (in seconds).

    Produced by :class:`PacketDecoder`, consumed by :class:`ColorConverter`. The
    handle wraps a raw pointer and is process-local. pts/duration are stamped by
    the decoder (which knows the stream time base) and carried here so the
    :class:`ColorConverter` need not be bound to any stream.

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
    ):
        self._handle = handle
        self._device = device
        self.pts_seconds = pts_seconds
        self.duration_seconds = duration_seconds

    # TODO_API_BREAKDOWN P2 Why is this a property???
    # Do we even need to havet this?
    @property
    def device(self) -> str:
        return self._device

    # TODO_API_BREAKDOWN P1: Really need to think hard about the API of *all* of
    # this.
    # materialize()?
    # What about the planes - should they be 2D (right now they are)? Should we
    # give them names?
    def materialize(self) -> RawFrame:
        (
            p0,
            p1,
            p2,
            p3,
            pix_fmt,
            colorspace,
            color_range,
            bit_depth,
        ) = _blocks_frame_to_planes(self._handle, self._device)
        # Absent components come back as empty tensors; real ones are 2D views.
        planes = tuple(p for p in (p0, p1, p2, p3) if p.numel() > 0)
        return RawFrame(
            planes=planes,
            pix_fmt=pix_fmt,
            colorspace=colorspace,
            color_range=color_range,
            bit_depth=bit_depth,
        )
