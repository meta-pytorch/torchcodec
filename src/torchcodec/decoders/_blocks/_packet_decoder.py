# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

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
from ._demuxer import _BaseDemuxer
from ._frame import Packet, RawAudioSamples, RawFrame


# TODO_API_BREAKDOWN DOC P1 revisit every single docstring / comments at some point.


class PacketDecoder:
    """Decode building block: turns compressed :class:`Packet`\\ s into decoded
    frames.

    What comes out follows the demuxer it was built from: (YUV)
    :class:`RawFrame`\\ s for a :class:`VideoDemuxer`, :class:`RawAudioSamples`
    for an :class:`AudioDemuxer`. Decoding is the same operation either way,
    which is why this is one block rather than two.

    Built from a demuxer (for its codec parameters) and stateful: it holds the
    codec's buffered state, which is reference frames for video and
    overlap-add state for a lossy audio codec. Passive and *not* thread-safe:
    use one ``PacketDecoder`` per thread. FFmpeg's internal codec thread count
    is kept at 1 for now (not exposed); parallelism comes from composing blocks
    on your own threads.

    ``device`` accepts a string or a ``torch.device``. It defaults to ``None``,
    which means the current default device (see ``torch.set_default_device``).
    Audio is decoded on the CPU only: passing a non-CPU ``device`` alongside an
    :class:`AudioDemuxer` raises, and leaving it unspecified gives CPU rather
    than picking up a non-CPU default.
    """

    def __init__(self, demuxer: _BaseDemuxer, device: str | torch.device | None = None):
        self._is_audio = demuxer._media_type == "audio"
        if self._is_audio:
            if device is not None and torch.device(device).type != "cpu":
                raise ValueError(
                    f"Got device={device}, but audio can only be decoded on the "
                    "CPU. Leave device unspecified, or pass 'cpu'."
                )
            device_str = "cpu"
        else:
            device_str = convert_device_to_str(device)

        self._handle = _blocks_create_packet_decoder(
            demuxer._handle, num_threads=1, device=device_str
        )
        self._drained = False

    def _receive_ready_video_frames(self) -> list[RawFrame]:
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

    def _receive_ready_audio_samples(self) -> list[RawAudioSamples]:
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

    def _receive_ready_frames(self) -> list[RawFrame] | list[RawAudioSamples]:
        if self._is_audio:
            return self._receive_ready_audio_samples()
        return self._receive_ready_video_frames()

    def decode(self, packet: Packet) -> list[RawFrame] | list[RawAudioSamples]:
        """Send one packet and return whatever is now ready (possibly empty,
        e.g. while the codec buffers B-frames)."""
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

    def drain(self) -> list[RawFrame] | list[RawAudioSamples]:
        _blocks_packet_decoder_send_eof(self._handle)
        frames = self._receive_ready_frames()
        self._drained = True
        return frames

    def reset(self) -> None:
        _blocks_packet_decoder_reset(self._handle)
        self._drained = False
