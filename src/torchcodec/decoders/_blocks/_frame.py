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


# TODO_API_BREAKDOWN P1 API design: should these fields (pix_format, colorspace
# etc.) also exist on the DecodedFrame class? Should there **just** be the
# DecodedFrame class and actually just call materialize() transparently whenever
# the user wants to access the planes? If materialize() is super cheap (it
# should be???) then this might be a good UX.
@dataclass
class RawFrame:
    planes: tuple[torch.Tensor, ...]
    pix_fmt: str  # FFmpeg pixel-format name, e.g. "yuv420p"
    colorspace: str  # e.g. "bt709"
    color_range: str  # "tv" (limited) or "pc" (full)
    # The depth of pix_fmt (always).
    # This is also the source's bit depth, except for 12b-bit sources CUDA
    # frames: those are technically P012, but P012 was only introduced in FFmpeg
    # 6. So For FFmpeg < 6, we must report those as P016, and so the bit_depth
    # field here reports 16 (on CPU, it'd still be 12).
    # Everything downstream still reads right, because those samples are
    # msb-aligned and are therefore genuinely valid 16-bit ones, with 4 zeroed
    # low bits.
    #
    # TODO_API_BREAKDOWN P2: We can't do anything about the P016 report, but
    # should this actually report the depth of the source instead of the depth
    # of the pixel format? Again the only discrepency arises for 12-bit sources
    # on CUDA for FFmpeg < 6.
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
