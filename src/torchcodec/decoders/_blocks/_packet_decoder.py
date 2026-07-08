# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchcodec._core.ops import (
    _blocks_create_packet_decoder,
    _blocks_packet_decoder_receive_frame,
    _blocks_packet_decoder_send_eof,
    _blocks_packet_decoder_send_packet,
)

from ._demuxer import Demuxer
from ._frame import DecodedFrame, Packet


# TODO_API_BREAKDOWN revisit every single docstring / comments at some point.


class PacketDecoder:
    """Decode building block: turns compressed :class:`Packet`\\ s into decoded
    (YUV) :class:`DecodedFrame`\\ s.

    Built from a :class:`Demuxer` (for its codec parameters) and stateful (it
    holds the codec's reference-frame buffer). Passive and *not* thread-safe:
    use one ``PacketDecoder`` per thread. FFmpeg's internal codec thread count
    is kept at 1 for now (not exposed); parallelism comes from composing blocks
    on your own threads.
    """

    def __init__(self, demuxer: Demuxer):
        self._handle = _blocks_create_packet_decoder(demuxer._handle, num_threads=1)

    def _drain(self) -> list[DecodedFrame]:
        frames = []
        while True:
            handle, status, pts_seconds, duration_seconds = (
                _blocks_packet_decoder_receive_frame(self._handle)
            )
            if status != 0:  # EAGAIN (need more packets) or EOF: nothing ready
                break
            frames.append(DecodedFrame(handle, pts_seconds, duration_seconds))
        return frames

    def decode(self, packet: Packet) -> list[DecodedFrame]:
        """Send one packet and return whatever frames are now ready (possibly
        empty, e.g. while the codec buffers B-frames)."""
        status = _blocks_packet_decoder_send_packet(self._handle, packet._handle)
        if status < 0:
            raise RuntimeError(f"Failed to send packet to decoder (status {status})")
        return self._drain()

    def flush(self) -> list[DecodedFrame]:
        """Signal end-of-stream and return all remaining buffered frames. Call
        once, after the last packet."""
        _blocks_packet_decoder_send_eof(self._handle)
        return self._drain()
