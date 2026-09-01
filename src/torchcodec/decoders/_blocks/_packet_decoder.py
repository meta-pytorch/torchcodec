# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Generic, TypeVar

import torch

from torchcodec._core.ops import (
    _blocks_audio_packet_decoder_receive_frame,
    _blocks_create_packet_decoder,
    _blocks_packet_decoder_receive_frame,
    _blocks_packet_decoder_reset,
    _blocks_packet_decoder_send_eof,
    _blocks_packet_decoder_send_packet,
)

from .._decoder_utils import convert_device_to_str
from ._demuxer import AudioDemuxer, VideoDemuxer
from ._frame import Packet, RawAudioSamples, RawFrame


# TODO_API_BREAKDOWN DOC P1 revisit every single docstring / comments at some point.

_Decoded = TypeVar("_Decoded", RawFrame, RawAudioSamples)


class _BasePacketDecoder(Generic[_Decoded]):
    """Shared machinery for :class:`VideoPacketDecoder` and
    :class:`AudioPacketDecoder`.

    Decoding is the same ``avcodec_send_packet`` / ``avcodec_receive_frame``
    pair for both, so the C++ side is a single class; what differs is only what
    a decoded frame is turned into, which is :meth:`_receive_ready_frames`.
    """

    def __init__(self, demuxer, device_str: str):
        self._handle = _blocks_create_packet_decoder(
            demuxer._handle, num_threads=1, device=device_str
        )
        self._drained = False

    def _receive_ready_frames(self) -> list[_Decoded]:
        raise NotImplementedError

    def decode(self, packet: Packet) -> list[_Decoded]:
        """Send one packet and return whatever is now ready (possibly empty,
        e.g. while the codec buffers B-frames)."""
        if packet.is_eof:
            raise ValueError(
                "This is an end-of-stream marker, not a packet: it carries no "
                "data to decode. Call drain() instead, to get the frames the "
                "codec is still holding on to."
            )
        if self._drained:
            raise RuntimeError(
                "This decoder has been drained, and a codec that has been told "
                "the stream ended ignores any further packet. Create a new "
                "decoder to decode another stream."
            )
        status = _blocks_packet_decoder_send_packet(self._handle, packet._handle)
        if status < 0:
            raise RuntimeError(f"Failed to send packet to decoder (status {status})")
        return self._receive_ready_frames()

    def drain(self) -> list[_Decoded]:
        """Tell the codec the stream ended, and return the frames it was still
        holding on to."""
        _blocks_packet_decoder_send_eof(self._handle)
        frames = self._receive_ready_frames()
        self._drained = True
        return frames

    def reset(self) -> None:
        """Drop the codec's buffered state and start over. Needed after the
        demuxer seeked, and after ``drain()``."""
        _blocks_packet_decoder_reset(self._handle)
        self._drained = False


class VideoPacketDecoder(_BasePacketDecoder[RawFrame]):
    """Decode building block: turns compressed :class:`Packet`\\ s into decoded
    (YUV) :class:`RawFrame`\\ s.

    Built from a :class:`VideoDemuxer` (for its codec parameters) and stateful
    (it holds the codec's reference-frame buffer). Passive and *not*
    thread-safe: use one ``VideoPacketDecoder`` per thread. FFmpeg's internal
    codec thread count is kept at 1 for now (not exposed); parallelism comes
    from composing blocks on your own threads.

    ``device`` accepts a string or a ``torch.device``. It defaults to ``None``,
    which means the current default device (see ``torch.set_default_device``).
    """

    def __init__(self, demuxer: VideoDemuxer, device: str | torch.device | None = None):
        super().__init__(demuxer, convert_device_to_str(device))

    def _receive_ready_frames(self) -> list[RawFrame]:
        frames = []
        while True:
            handle, status, pts_seconds, duration_seconds, device, storage = (
                _blocks_packet_decoder_receive_frame(self._handle)
            )
            if status != 0:  # EAGAIN (need more packets) or EOF: nothing ready
                break
            frames.append(
                RawFrame(
                    handle,
                    pts_seconds,
                    duration_seconds,
                    device=device,
                    storage=storage if storage.numel() > 0 else None,
                )
            )
        return frames


class AudioPacketDecoder(_BasePacketDecoder[RawAudioSamples]):
    """Decode building block: turns compressed :class:`Packet`\\ s into
    :class:`RawAudioSamples`.

    Built from an :class:`AudioDemuxer` (for its codec parameters) and stateful:
    a lossy codec's overlap-add state means the frames decoded right after a
    seek are subtly wrong until it re-primes, so ``reset()`` is necessary but
    not by itself sufficient - see :meth:`AudioDemuxer.seek`. Passive and *not*
    thread-safe: use one ``AudioPacketDecoder`` per thread.

    There is no ``device`` parameter: audio is always decoded on the CPU, and
    that doesn't change with ``torch.set_default_device``.
    """

    def __init__(self, demuxer: AudioDemuxer):
        super().__init__(demuxer, "cpu")

    def _receive_ready_frames(self) -> list[RawAudioSamples]:
        samples = []
        while True:
            data, status, pts_seconds, duration_seconds, sample_rate, sample_format = (
                _blocks_audio_packet_decoder_receive_frame(self._handle)
            )
            if status != 0:  # EAGAIN (need more packets) or EOF: nothing ready
                break
            samples.append(
                RawAudioSamples(
                    data=data,
                    sample_rate=sample_rate,
                    sample_format=sample_format,
                    pts_seconds=pts_seconds,
                    duration_seconds=duration_seconds,
                )
            )
        return samples
