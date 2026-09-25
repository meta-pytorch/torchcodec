# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchcodec._core.ops import (
    _blocks_audio_converter_convert,
    _blocks_audio_converter_drain,
    _blocks_audio_converter_reset,
    _blocks_create_audio_converter,
)
from torchcodec._frame import AudioSamples

from ._frame import RawAudioSamples
from ._helpers import _process_local


@_process_local(
    "A resampler carries filter history across calls that FFmpeg cannot "
    "serialise. Construct one in each process."
)
class AudioConverter:
    """Turn :class:`RawAudioSamples` into normalised float32
    :class:`~torchcodec.AudioSamples`, optionally resampling and remixing
    channels.

    Low-level API: for straightforward decoding, use
    :class:`~torchcodec.decoders.AudioDecoder` instead.

    .. code-block:: python

        converter = AudioConverter(sample_rate=16_000)

        for packet in demuxer:
            for raw_samples in packet_decoder.decode(packet):
                samples = converter.convert(raw_samples)
        for raw_samples in packet_decoder.drain():
            samples = converter.convert(raw_samples)
        samples = converter.drain()

    Unlike a :class:`ColorConverter`, this object is a stateful stream
    processor, and it is bound to an audio stream: feed it the stream's samples
    in order. When resampling, it holds samples back between calls, so you won't
    necessarily get the same number of samples out as you put in for a given
    call to :meth:`convert`.

    Args:
        sample_rate (int, optional): The output sample rate. Defaults to the
            source's own, i.e. no resampling.
        num_channels (int, optional): The output number of channels. Defaults
            to the source's own.
    """

    def __init__(self, sample_rate: int | None = None, num_channels: int | None = None):
        self._handle = _blocks_create_audio_converter(
            sample_rate=sample_rate, num_channels=num_channels
        )
        self._requested_sample_rate = sample_rate
        self._drained = False

        self._first_frame_pts_seconds: float | None = None
        self._out_sample_rate: int | None = None
        self._num_emitted_samples = 0
        # The demuxer position these samples come from. See Packet._generation:
        # a resampler carries state across calls, so a seek invalidates it just
        # as it invalidates the decoder's.
        self._generation: int | None = None

    def _wrap(self, data) -> AudioSamples:
        assert self._out_sample_rate is not None  # mypy
        assert self._first_frame_pts_seconds is not None  # mypy
        # We manually compute the pts of all frames but the first one: when
        # resampling happens, libswresample buffers samples and emits them
        # later. Not all the samples of the first frame are necessarily emitted
        # immediately, they may be emitted with the next frame. Without this,
        # we'd fail the test_audio_converter_pts_is_contiguous() test.
        pts_seconds = (
            self._first_frame_pts_seconds
            + self._num_emitted_samples / self._out_sample_rate
        )
        self._num_emitted_samples += data.shape[1]
        return AudioSamples(
            data=data,
            pts_seconds=pts_seconds,
            duration_seconds=data.shape[1] / self._out_sample_rate,
            sample_rate=self._out_sample_rate,
        )

    def convert(self, raw_samples: RawAudioSamples) -> AudioSamples:
        """Convert one :class:`RawAudioSamples` into normalised float32
        :class:`~torchcodec.AudioSamples`.

        You may not get the same number of samples out as you put in, especially
        if you are resampling.

        Returns:
            The converted samples, normalised float32 in ``[-1, 1]``. When
            resampling, fewer than were passed in, possibly none.
        """
        if self._generation is None:
            self._generation = raw_samples._generation
        elif self._generation != raw_samples._generation:
            raise RuntimeError(
                "The demuxer seeked since this converter was last reset(), and "
                "a resampler carries state across calls, so these samples "
                "would be resampled against the wrong history. Call reset() on "
                "every converter fed by that demuxer after a seek."
            )
        if self._drained:
            raise RuntimeError(
                "This AudioConverter has been drained. Call reset() to convert "
                "more samples."
            )
        if self._first_frame_pts_seconds is None:
            self._first_frame_pts_seconds = raw_samples.pts_seconds
            self._out_sample_rate = (
                self._requested_sample_rate
                if self._requested_sample_rate is not None
                else raw_samples.sample_rate
            )
        data = _blocks_audio_converter_convert(
            self._handle, raw_samples.data, raw_samples.sample_rate
        )
        return self._wrap(data)

    def drain(self) -> AudioSamples:
        """Return the samples the resampler was still holding on to.

        Empty unless you are resampling. This is not the codec's own buffer,
        which :meth:`AudioPacketDecoder.drain` takes care of.
        """
        if self._first_frame_pts_seconds is None:
            raise RuntimeError(
                "This AudioConverter hasn't converted any samples, so there is "
                "nothing to drain."
            )
        data = _blocks_audio_converter_drain(self._handle)
        self._drained = True
        return self._wrap(data)

    def reset(self) -> None:
        """Drop the resampler's state and start over.

        Needed after a :meth:`Demuxer.seek`, after :meth:`drain`, and before
        converting a different stream.
        """
        _blocks_audio_converter_reset(self._handle)
        self._drained = False
        self._first_frame_pts_seconds = None
        self._out_sample_rate = None
        self._generation = None
        self._num_emitted_samples = 0
