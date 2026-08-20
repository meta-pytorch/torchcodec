# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from torchcodec._core.ops import (
    _blocks_create_packet_decoder,
    _blocks_packet_decoder_receive_frame,
    _blocks_packet_decoder_reset,
    _blocks_packet_decoder_send_eof,
    _blocks_packet_decoder_send_packet,
)

from .._decoder_utils import convert_device_to_str
from ._demuxer import Demuxer
from ._frame import DecodedFrame, Packet


# TODO_API_BREAKDOWN DOC P1 revisit every single docstring / comments at some point.


class PacketDecoder:
    """Decode building block: turns compressed :class:`Packet`\\ s into decoded
    (YUV) :class:`DecodedFrame`\\ s.

    Built from a :class:`Demuxer` (for its codec parameters) and stateful (it
    holds the codec's reference-frame buffer). Passive and *not* thread-safe:
    use one ``PacketDecoder`` per thread. FFmpeg's internal codec thread count
    is kept at 1 for now (not exposed); parallelism comes from composing blocks
    on your own threads.

    ``device`` accepts a string or a ``torch.device``. It defaults to ``None``,
    which means the current default device (see ``torch.set_default_device``).
    """

    def __init__(self, demuxer: Demuxer, device: str | torch.device | None = None):
        self._handle = _blocks_create_packet_decoder(
            demuxer._handle, num_threads=1, device=convert_device_to_str(device)
        )
        self._drained = False

    def _receive_ready_frames(self) -> list[DecodedFrame]:
        frames = []
        while True:
            handle, status, pts_seconds, duration_seconds, device, storage = (
                _blocks_packet_decoder_receive_frame(self._handle)
            )
            if status != 0:  # EAGAIN (need more packets) or EOF: nothing ready
                break
            frames.append(
                DecodedFrame(
                    handle,
                    pts_seconds,
                    duration_seconds,
                    device=device,
                    storage=storage if storage.numel() > 0 else None,
                )
            )
        return frames

    def decode(self, packet: Packet) -> list[DecodedFrame]:
        """Send one packet and return whatever frames are now ready (possibly
        empty, e.g. while the codec buffers B-frames)."""
        if self._drained:
            raise RuntimeError(
                "This PacketDecoder has been drained, and a codec that has been "
                "told the stream ended ignores any further packet. Create a new "
                "PacketDecoder to decode another stream."
            )
        status = _blocks_packet_decoder_send_packet(self._handle, packet._handle)
        if status < 0:
            raise RuntimeError(f"Failed to send packet to decoder (status {status})")
        return self._receive_ready_frames()

    def drain(self) -> list[DecodedFrame]:
        _blocks_packet_decoder_send_eof(self._handle)
        frames = self._receive_ready_frames()
        self._drained = True
        return frames

    def reset(self) -> None:
        _blocks_packet_decoder_reset(self._handle)
        self._drained = False
