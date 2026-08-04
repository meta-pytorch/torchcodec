# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings

import torch

from torchcodec._core.ops import _blocks_packet_from_tensor


class Packet:
    """Opaque, thread-movable handle to a demuxed (compressed) packet.

    Produced by :class:`Demuxer` or built from raw bytes with
    :meth:`from_tensor` / :meth:`from_bytes`, consumed by :class:`PacketDecoder`.

    It wraps a raw pointer, so it is only valid within the process that created it.
    """

    def __init__(self, handle: torch.Tensor):
        self._handle = handle

    @classmethod
    def from_tensor(
        cls,
        data: torch.Tensor,
        *,
        pts: int,
        duration: int,
        is_key_frame: bool,
    ) -> Packet:
        """Build a :class:`Packet` from raw compressed bytes.

        ``data`` is a 1D uint8 tensor holding the compressed packet payload.
        ``pts`` and ``duration`` are expressed in the time base of the stream the
        consuming :class:`PacketDecoder` was created for.
        The packet's dts is set to ``pts``, which is only correct for streams without frame
        reordering (no B-frames).
        """
        return cls(
            _blocks_packet_from_tensor(
                data, pts=pts, duration=duration, is_key_frame=is_key_frame
            )
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        pts: int,
        duration: int,
        is_key_frame: bool,
    ) -> Packet:
        """Build a :class:`Packet` from raw compressed bytes.

        Same as :meth:`from_tensor`, but takes the payload as ``bytes``.
        """
        with warnings.catch_warnings():
            # Ignore warning stating that the underlying data buffer is non-writable.
            warnings.filterwarnings("ignore", category=UserWarning)
            buffer = torch.frombuffer(data, dtype=torch.uint8)
        return cls.from_tensor(
            buffer, pts=pts, duration=duration, is_key_frame=is_key_frame
        )


class DecodedFrame:
    """A decoded (YUV) frame: an opaque, thread-movable handle to the raw frame
    plus its presentation timestamp and duration (in seconds).

    Produced by :class:`PacketDecoder`, consumed by :class:`ColorConverter`. The
    handle wraps a raw pointer and is process-local. pts/duration are stamped by
    the decoder (which knows the stream time base) and carried here so the
    :class:`ColorConverter` need not be bound to any stream.
    """

    def __init__(
        self,
        handle: torch.Tensor,
        pts_seconds: float,
        duration_seconds: float,
    ):
        self._handle = handle
        self.pts_seconds = pts_seconds
        self.duration_seconds = duration_seconds
