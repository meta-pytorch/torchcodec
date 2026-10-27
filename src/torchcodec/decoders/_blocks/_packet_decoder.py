# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fractions import Fraction
from typing import Generic, TypeVar

import torch

from torchcodec._core.ops import (
    _blocks_audio_packet_decoder_receive_frame,
    _blocks_create_packet_decoder,
    _blocks_packet_decoder_receive_frame,
    _blocks_packet_decoder_reset,
    _blocks_packet_decoder_send_eof,
    _blocks_packet_decoder_send_packet,
    _blocks_packet_decoder_time_base,
)

from .._decoder_utils import convert_device_to_str
from ._decoder_parameters import (
    _DecoderParameters,
    AudioPacketDecoderParameters,
    VideoPacketDecoderParameters,
)
from ._frame import _NO_PTS, Packet, RawAudioSamples, RawFrame
from ._helpers import _process_local


# TODO_API_BREAKDOWN DOC P1 revisit every single comment across all low-level APIs

_Decoded = TypeVar("_Decoded", RawFrame, RawAudioSamples)


@_process_local(
    "A codec context holds reference frames and entropy state that FFmpeg "
    "cannot serialise. Build a decoder in each process instead, from the same "
    "decoder parameters: those can be pickled."
)
class _BasePacketDecoder(Generic[_Decoded]):
    _handle: torch.Tensor
    _drained: bool
    _generation: int | None

    def _init(
        self,
        parameters: _DecoderParameters,
        device_str: str,
        num_ffmpeg_threads: int,
        decoder: str | None,
    ) -> None:
        self._handle = _blocks_create_packet_decoder(
            parameters._to_strings(),
            num_threads=num_ffmpeg_threads,
            device=device_str,
            decoder=decoder,
        )
        self._drained = False
        # The demuxer position these packets come from. None until the first
        # packet, and again after every reset(), so it is adopted rather than
        # tracked: the decoder never needs a reference back to the demuxer.
        self._generation = None

    @property
    def time_base(self) -> Fraction:
        """The unit of the integer :term:`pts` of the frames this decoder
        produces: a pts of ``n`` is ``n * time_base`` seconds."""
        return Fraction(*_blocks_packet_decoder_time_base(self._handle))

    def _receive_ready_frames(self) -> list[_Decoded]:
        raise NotImplementedError

    def decode(self, packet: Packet) -> list[_Decoded]:
        if self._drained:
            raise RuntimeError(
                "This decoder has been drained, and a codec that has been told "
                "the stream ended ignores any further packet. Create a new "
                "decoder to decode another stream."
            )
        if self._generation is None:
            self._generation = packet._generation
        elif self._generation != packet._generation:
            raise RuntimeError(
                "The demuxer seeked since this decoder was last reset(), so "
                "this packet is from a position the codec knows nothing about "
                "- decoding it would produce plausible-looking garbage. Call "
                "reset() on every decoder fed by that demuxer after a seek."
            )
        status = _blocks_packet_decoder_send_packet(self._handle, packet._handle)
        if status < 0:
            raise RuntimeError(f"Failed to send packet to decoder (status {status})")
        return self._receive_ready_frames()

    def drain(self) -> list[_Decoded]:
        _blocks_packet_decoder_send_eof(self._handle)
        frames = self._receive_ready_frames()
        self._drained = True
        return frames

    def reset(self) -> None:
        """Drop the codec's buffered state and start over.

        Needed after a :meth:`Demuxer.seek`, and after :meth:`drain`.
        """
        _blocks_packet_decoder_reset(self._handle)
        self._drained = False
        self._generation = None


class VideoPacketDecoder(_BasePacketDecoder[RawFrame]):
    """Decodes the compressed :class:`Packet`\\ s of one video stream into
    :class:`RawFrame`\\ s.

    This is a low-level API: for straightforward decoding, use
    :class:`~torchcodec.decoders.VideoDecoder` instead.

    It is built from the :class:`VideoPacketDecoderParameters` describing the
    stream's packets. :meth:`VideoStream.make_decoder` builds one from
    :attr:`VideoStream.decoder_parameters`.

    Feed it one packet at a time, and drain it at the end::

        demuxer = Demuxer("video.mp4")
        decoder = demuxer.streams[0].make_decoder()

        for packet in demuxer:
            for raw_frame in decoder.decode(packet):
                ...
        for raw_frame in decoder.drain():
            ...

    It is stateful. It holds the codec's reference-frame buffer, so it expects
    the packets of its own stream, in decoding order: the order a demuxer
    produces them in.

    Args:
        parameters (VideoPacketDecoderParameters): Describes the packets this
            decoder decodes.
        device (str or torch.device, optional): The device to decode on (cpu or
            CUDA). Frames come out on this device. If ``None`` (default), the
            current default device is used (see ``torch.set_default_device``).
        num_ffmpeg_threads (int, optional): The number of threads to use for
            CPU decoding. This has no effect when decoding on GPU. Use 1 for
            single-threaded decoding, which may be best if you are decoding
            multiple streams in parallel. Use a higher number for
            multi-threaded decoding, which is best for a single stream. Passing
            0 lets FFmpeg decide on the number of threads. Default: 1.
        decoder (str, optional): The name of the FFmpeg decoder to use, e.g.
            ``"libdav1d"``. It must decode the codec of ``parameters``. By
            default FFmpeg picks one, or the device does.
    """

    def __init__(
        self,
        parameters: VideoPacketDecoderParameters,
        device: str | torch.device | None = None,
        *,
        num_ffmpeg_threads: int = 1,
        decoder: str | None = None,
    ):
        if not isinstance(parameters, VideoPacketDecoderParameters):
            raise TypeError(
                "parameters must be a VideoPacketDecoderParameters, got "
                f"{type(parameters).__name__}."
            )
        if num_ffmpeg_threads is None:
            raise ValueError(f"{num_ffmpeg_threads = } should be an int.")
        self._init(
            parameters, convert_device_to_str(device), num_ffmpeg_threads, decoder
        )

    # methods calling super() only to pin the return type down to RawFrame. The
    # base class is generic over _Decoded, which isn't ideal for the rendered
    # docs.
    def decode(self, packet: Packet) -> list[RawFrame]:
        """Send one :class:`Packet` to the codec and return the
        :class:`RawFrame`\\ s that are ready.

        **This can return zero, one, or more than one** :class:`RawFrame`. What
        comes back is not the decoding of the packet you just passed: a codec
        that is buffering B-frames, or still priming itself, will emit what it owes
        you on a later call.

        Args:
            packet (Packet): A packet of this decoder's own stream.

        Returns:
            The possibly empty list of :class:`RawFrame`\\ s that the codec has
            ready, in presentation order.

        Raises:
            RuntimeError: If this decoder has been drained, or if the demuxer
                seeked without it being :meth:`reset` afterwards.
        """
        return super().decode(packet)

    def drain(self) -> list[RawFrame]:
        """Tell the codec the stream has ended, and return the
        :class:`RawFrame`\\ s it was still holding.

        Skipping this loses the tail of the stream. A drained decoder refuses
        any further packet; :meth:`reset` makes it usable again.

        Returns:
            The possibly empty list of :class:`RawFrame`\\ s that the codec was
            still holding, in presentation order.
        """
        return super().drain()

    def _receive_ready_frames(self) -> list[RawFrame]:
        frames = []
        while True:
            handle, status, pts, pts_seconds, duration_seconds, storage = (
                _blocks_packet_decoder_receive_frame(self._handle)
            )
            if status != 0:  # EAGAIN (need more packets) or EOF: nothing ready
                break
            frames.append(
                RawFrame(
                    handle,
                    None if pts == _NO_PTS else pts,
                    pts_seconds,
                    duration_seconds,
                    storage=storage if storage.numel() > 0 else None,
                )
            )
        return frames


class AudioPacketDecoder(_BasePacketDecoder[RawAudioSamples]):
    """Decodes the compressed :class:`Packet`\\ s of one audio stream into
    :class:`RawAudioSamples`.

    This is a low-level API: for straightforward decoding, use
    :class:`~torchcodec.decoders.AudioDecoder` instead.

    It is built from the :class:`AudioPacketDecoderParameters` describing the
    stream's packets. :meth:`AudioStream.make_decoder` builds one from
    :attr:`AudioStream.decoder_parameters`. Audio is always decoded on the CPU.

    Feed it one packet at a time, and drain it at the end::

        demuxer = Demuxer("audio.mp3", streams="audio")
        decoder = demuxer.streams[0].make_decoder()

        for packet in demuxer:
            for raw_samples in decoder.decode(packet):
                ...
        for raw_samples in decoder.drain():
            ...

    It is stateful. A lossy codec carries state from one frame to the next, so
    it expects the packets of its own stream, in the order the demuxer produced
    them. After a :meth:`Demuxer.seek`, :meth:`reset` is necessary but not
    sufficient: the first samples that come out are subtly wrong until the codec
    re-primes, so decode a margin before your target and throw it away.

    Args:
        parameters (AudioPacketDecoderParameters): Describes the packets this
            decoder decodes.
        decoder (str, optional): The name of the FFmpeg decoder to use, e.g.
            ``"mp3float"``. It must decode the codec of ``parameters``. By
            default FFmpeg picks one.
    """

    def __init__(
        self,
        parameters: AudioPacketDecoderParameters,
        *,
        decoder: str | None = None,
    ):
        if not isinstance(parameters, AudioPacketDecoderParameters):
            raise TypeError(
                "parameters must be an AudioPacketDecoderParameters, got "
                f"{type(parameters).__name__}."
            )
        self._init(parameters, "cpu", num_ffmpeg_threads=1, decoder=decoder)

    # See VideoPacketDecoder: pinning the return type down to RawAudioSamples.
    def decode(self, packet: Packet) -> list[RawAudioSamples]:
        """Send one :class:`Packet` to the codec and return the
        :class:`RawAudioSamples` that are ready.

        **This can return zero, one, or more than one**
        :class:`RawAudioSamples`. What comes back is not the decoding of the
        packet you just passed: a codec that is still priming itself will emit
        what it owes you on a later call.

        Args:
            packet (Packet): A packet of this decoder's own stream.

        Returns:
            The possibly empty list of :class:`RawAudioSamples` that the codec
            has ready, in presentation order.

        Raises:
            RuntimeError: If this decoder has been drained, or if the demuxer
                seeked without it being :meth:`reset` afterwards.
        """
        return super().decode(packet)

    def drain(self) -> list[RawAudioSamples]:
        """Tell the codec the stream has ended, and return the
        :class:`RawAudioSamples` it was still holding.

        Skipping this loses the tail of the stream. A drained decoder refuses
        any further packet; :meth:`reset` makes it usable again.

        Returns:
            The possibly empty list of :class:`RawAudioSamples` that the codec
            was still holding, in presentation order.
        """
        return super().drain()

    def _receive_ready_frames(self) -> list[RawAudioSamples]:
        samples = []
        while True:
            data, status, pts, pts_seconds, duration_seconds, sample_rate, _ = (
                _blocks_audio_packet_decoder_receive_frame(self._handle)
            )
            if status != 0:  # EAGAIN (need more packets) or EOF: nothing ready
                break
            samples.append(
                RawAudioSamples(
                    data=data,
                    sample_rate=sample_rate,
                    pts=None if pts == _NO_PTS else pts,
                    pts_seconds=pts_seconds,
                    duration_seconds=duration_seconds,
                    # Carried onward so AudioConverter can make the same check:
                    # a seek invalidates the resampler's state too.
                    _generation=self._generation or 0,
                )
            )
        return samples
