# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Generic, TYPE_CHECKING, TypeVar

import torch

from torchcodec._core.ops import (
    _blocks_audio_packet_decoder_receive_frame,
    _blocks_create_packet_decoder,
    _blocks_packet_decoder_receive_frame,
    _blocks_packet_decoder_reset,
    _blocks_packet_decoder_send_eof,
    _blocks_packet_decoder_send_packet,
)

from ._frame import Packet, RawAudioSamples, RawFrame

if TYPE_CHECKING:
    # Only for the annotation: _demuxer imports this module to build decoders,
    # so importing it back at runtime would be circular.
    from ._demuxer import _Stream


# TODO_API_BREAKDOWN DOC P1 revisit every single docstring / comments at some point.

_Decoded = TypeVar("_Decoded", RawFrame, RawAudioSamples)
_Self = TypeVar("_Self", bound="_BasePacketDecoder")


class _BasePacketDecoder(Generic[_Decoded]):
    _handle: torch.Tensor
    _drained: bool
    _generation: int | None

    # *args so that a call with arguments gets the message below rather than a
    # TypeError about the argument count.
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__} cannot be instantiated directly. Build one "
            "from the stream whose packets it decodes, with "
            "stream.make_decoder()."
        )

    @classmethod
    def _from_stream(cls: type[_Self], stream: _Stream, device_str: str) -> _Self:
        decoder = cls.__new__(cls)
        decoder._handle = _blocks_create_packet_decoder(
            stream._demuxer._handle,
            stream_index=stream.index,
            num_threads=1,
            device=device_str,
        )
        decoder._drained = False
        # The demuxer position these packets come from. None until the first
        # packet, and again after every reset(), so it is adopted rather than
        # tracked: the decoder never needs a reference back to the demuxer.
        decoder._generation = None
        return decoder

    def _receive_ready_frames(self) -> list[_Decoded]:
        raise NotImplementedError

    def decode(self, packet: Packet) -> list[_Decoded]:
        """Send one :class:`Packet` to the codec and return whatever is ready.

        The result is often empty, and it is not "the decoding of that packet":
        a codec that is buffering B-frames, or still priming itself, emits what
        it owes you on a later call.

        Args:
            packet (Packet): A packet of this decoder's own stream.

        Returns:
            What the codec had ready, in presentation order. Possibly nothing.

        Raises:
            RuntimeError: If this decoder has been drained, or if the demuxer
                seeked without it being :meth:`reset` afterwards.
        """
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
        """Tell the codec the stream has ended, and return what it was still
        holding.

        Skipping this loses the tail of the stream. A drained decoder refuses
        any further packet; :meth:`reset` makes it usable again.

        Returns:
            The last of what the codec had buffered, in presentation order.
        """
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

    You should not build one yourself: :meth:`VideoStream.make_decoder` is what
    creates it. Frames come out on the device given there.

    It is stateful. It holds the codec's reference-frame buffer, so it expects
    the packets of its own stream, in the order the demuxer produced them.
    """

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
            handle, status, pts_seconds, duration_seconds, storage = (
                _blocks_packet_decoder_receive_frame(self._handle)
            )
            if status != 0:  # EAGAIN (need more packets) or EOF: nothing ready
                break
            frames.append(
                RawFrame(
                    handle,
                    pts_seconds,
                    duration_seconds,
                    storage=storage if storage.numel() > 0 else None,
                )
            )
        return frames


class AudioPacketDecoder(_BasePacketDecoder[RawAudioSamples]):
    """TODO_API_BREAKDOWN DOC"""

    # See VideoPacketDecoder: pinning the return type down to RawAudioSamples.
    def decode(self, packet: Packet) -> list[RawAudioSamples]:
        return super().decode(packet)

    def drain(self) -> list[RawAudioSamples]:
        return super().drain()

    def _receive_ready_frames(self) -> list[RawAudioSamples]:
        samples = []
        while True:
            data, status, pts_seconds, duration_seconds, sample_rate, _ = (
                _blocks_audio_packet_decoder_receive_frame(self._handle)
            )
            if status != 0:  # EAGAIN (need more packets) or EOF: nothing ready
                break
            samples.append(
                RawAudioSamples(
                    data=data,
                    sample_rate=sample_rate,
                    pts_seconds=pts_seconds,
                    duration_seconds=duration_seconds,
                    # Carried onward so AudioConverter can make the same check:
                    # a seek invalidates the resampler's state too.
                    _generation=self._generation or 0,
                )
            )
        return samples
