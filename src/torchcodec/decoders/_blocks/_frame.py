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


# TODO_API_BREAKDOWN DESIGN P1: should these fields (pix_format, colorspace
# etc.) also exist on the DecodedFrame class? Should there **just** be the
# DecodedFrame class and actually just call materialize() transparently whenever
# the user wants to access the planes? If materialize() is super cheap (it
# should be???) then this might be a good UX.
@dataclass
class RawFrame:
    planes: tuple[torch.Tensor, ...]
    # FFmpeg pixel-format name. On CPU this is the source's own format, e.g.
    # "yuv420p". On CUDA it is always an NVDEC surface format: "nv12",
    # "p010le", "p012le", "p016le", "yuv444p" or "yuv444p16le".
    pix_fmt: str
    colorspace: str  # e.g. "bt709"
    color_range: str  # "tv" (limited) or "pc" (full)
    # The depth of pix_fmt (always), which is the source's bit depth except
    # where a CUDA surface format's container is wider than the source samples:
    # a 10-bit 4:4:4 source is uploaded as yuv444p16le, and a 12-bit source is
    # tagged p016le on FFmpeg < 6 (which lacks p012le). Both report 16 here (on
    # CPU they'd report 10 and 12).
    # Everything downstream still reads right, because those samples are
    # msb-aligned and are therefore genuinely valid 16-bit ones, with zeroed
    # low bits.
    #
    # TODO_API_BREAKDOWN DESIGN P2: should this report the depth of the source instead
    # of the depth of the pixel format?
    bit_depth: int


# TODO_API_BREAKDOWN DESIGN P1: API design - especially the materialize() method but
# also the public fields, class name etc.
class DecodedFrame:
    """A decoded (YUV) frame: an opaque, thread-movable handle to the raw frame
    plus its presentation timestamp and duration (in seconds).

    Produced by :class:`PacketDecoder`, consumed by :class:`ColorConverter`. The
    handle wraps a raw pointer and is process-local. pts/duration are stamped by
    the decoder (which knows the stream time base) and carried here so the
    :class:`ColorConverter` need not be bound to any stream.

    ``device`` is where this frame's samples are, and it is always the device
    its decoder was created with. A CUDA decoder falls back to CPU decoding for
    streams NVDEC can't handle, but it uploads those frames before handing them
    out, so they are indistinguishable from NVDEC ones here.
    """

    def __init__(
        self,
        handle: torch.Tensor,
        pts_seconds: float,
        duration_seconds: float,
        device: str = "cpu",
        storage: torch.Tensor | None = None,
    ):
        self._handle = handle
        self._device = device
        #: The tensor owning this frame's GPU buffer, or ``None`` on CPU.
        #:
        #: All of :meth:`materialize`'s planes point into it. It comes from
        #: PyTorch's caching allocator, on the stream the decoder was running
        #: on, and the allocator only tracks that one stream. So if you read
        #: this frame's samples from a *different* CUDA stream and then drop
        #: the frame while that work is still queued, the allocator will hand
        #: the buffer to the decoder's next frame and overwrite it mid-read -
        #: silently, with no error.
        #:
        #: Tell the allocator about the read before dropping the frame::
        #:
        #:     with torch.cuda.stream(my_stream):
        #:         planes = frame.materialize().planes
        #:         result = my_kernel(planes)     # only *enqueued* here
        #:         frame.storage.record_stream(my_stream)
        #:     del frame
        #:
        #: You don't need this if you keep the frame (or its planes) alive
        #: until the work has run, or if you read it on the same stream the
        #: decoder used - which includes the default stream, shared by all
        #: threads. ``record_stream()`` on a plane does *not* work: the planes
        #: are not allocator-backed, so the call is silently ignored.
        self.storage = storage
        self.pts_seconds = pts_seconds
        self.duration_seconds = duration_seconds

    # TODO_API_BREAKDOWN DESIGN P2 Why is this a property???
    # Do we even need to havet this?
    @property
    def device(self) -> str:
        return self._device

    # TODO_API_BREAKDOWN DESIGN P1: Really need to think hard about the API of *all* of
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
