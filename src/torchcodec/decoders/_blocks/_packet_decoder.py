# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchcodec._core.ops import (
    _blocks_create_packet_decoder,
    _blocks_packet_decoder_receive_frame,
    _blocks_packet_decoder_reset,
    _blocks_packet_decoder_send_eof,
    _blocks_packet_decoder_send_packet,
)

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
    """

    # TODO_API_BREAKDOWN UF P1: device default should be None, here and everywhere else
    def __init__(self, demuxer: Demuxer, device="cpu"):
        self._handle = _blocks_create_packet_decoder(
            demuxer._handle, num_threads=1, device=device
        )

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
        status = _blocks_packet_decoder_send_packet(self._handle, packet._handle)
        if status < 0:
            raise RuntimeError(f"Failed to send packet to decoder (status {status})")
        return self._receive_ready_frames()

    def drain(self) -> list[DecodedFrame]:
        """Signal end-of-stream and return all remaining buffered frames. Call
        once, after the last packet.

        A decoder holds a few frames back (it needs later packets to
        reconstruct earlier ones), and this is what gets them out - what FFmpeg
        calls draining. The decoder is left ready to decode again, so it can be
        fed another stream without an explicit :meth:`reset`.
        """
        _blocks_packet_decoder_send_eof(self._handle)
        frames = self._receive_ready_frames()
        # Once it's been told the stream ended, a codec stays in that state
        # until its buffers are reset, and would ignore any further packet.
        _blocks_packet_decoder_reset(self._handle)
        return frames

    def reset(self) -> None:
        """Drop the buffered decoding state and start over.

        Call this after :meth:`Demuxer.seek`: the frames the codec holds as
        references belong to wherever we were before the seek, and decoding the
        packets from the new position against them produces corrupt output.
        Any :class:`DecodedFrame` already handed out stays valid.

        This is *not* :meth:`drain`: the frames still buffered here are
        discarded rather than returned. FFmpeg confusingly calls both of them
        flushing.
        """
        _blocks_packet_decoder_reset(self._handle)
