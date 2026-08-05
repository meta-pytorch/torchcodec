# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

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

    @property
    def device(self) -> str:
        return self._device

    def materialize(self) -> tuple[tuple[torch.Tensor, ...], str, int, int]:
        """Expose this frame's own samples as zero-copy views, before any color
        conversion.

        Returns ``(planes, pix_fmt, colorspace, color_range)``. ``planes`` holds
        one strided view per component, in the frame's native order: ``(Y, U, V)``
        for the common YUV formats (yuv420p, yuv444p, nv12, p016...), ``(R, G, B)``
        for RGB codecs, ``(Y,)`` for grayscale, plus a trailing alpha view when
        the format has one. Semi-planar (nv12, p016) and packed layouts come back
        as clean separate components -- the interleaving is hidden in the view's
        sample stride -- so 4:2:0 chroma views are half-resolution and are often
        non-contiguous. ``.contiguous()`` is the only thing that ever copies.

        Nothing is copied, so the samples are exactly what the codec produced,
        and 10-/12-bit (HDR) content keeps its full precision as uint16 views.
        The views live on :attr:`device` (a CUDA frame's samples stay on the
        GPU) and keep the frame alive, so they stay valid after this
        :class:`DecodedFrame` is dropped.

        ``pix_fmt`` is the FFmpeg pixel-format name (e.g. ``"yuv420p"``, or
        ``"nv12"`` / ``"p010le"`` for NVDEC-decoded frames). ``colorspace`` and
        ``color_range`` are the frame's ``AVColorSpace`` / ``AVColorRange`` enum
        values, for callers doing their own color math.
        """
        p0, p1, p2, p3, pix_fmt, colorspace, color_range = _blocks_frame_to_planes(
            self._handle, self._device
        )
        # Absent components come back as empty tensors; real ones are 2D views.
        planes = tuple(p for p in (p0, p1, p2, p3) if p.numel() > 0)
        return planes, pix_fmt, colorspace, color_range
