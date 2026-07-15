# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import io
import json
from pathlib import Path

import torch
from torchcodec import _core, AudioSamples
from torchcodec._core._decoder_utils import create_wav_decoder
from torchcodec._core._metadata import AudioStreamMetadata


class WavDecoder:
    """A fast decoder for WAV audio files.

    This is a lightweight, high-performance alternative to
    :class:`~torchcodec.decoders.AudioDecoder` that is specialized for WAV
    files. See :ref:`sphx_glr_generated_examples_decoding_performance_tips.py`
    for more details.

    Unlike :class:`~torchcodec.decoders.AudioDecoder`, this decoder does not
    support resampling (``sample_rate`` parameter) or channel remixing
    (``num_channels`` parameter). If you need those features, use
    :class:`~torchcodec.decoders.AudioDecoder`.

    Returned samples are float samples normalized in [-1, 1].

    Args:
        source (str, ``Pathlib.path``, bytes, ``torch.Tensor`` or file-like
            object): The source of the audio:

            - If ``str``: a path to a local WAV file.
            - If ``Pathlib.path``: a path to a local WAV file.
            - If ``bytes`` object or ``torch.Tensor``: the raw WAV data.
            - If file-like object: we read audio data from the object on
              demand. The object must expose the methods ``read(self, size:
              int) -> bytes`` and ``seek(self, offset: int, whence: int) ->
              int``.

    Attributes:
        metadata (AudioStreamMetadata): Metadata of the audio stream.
        stream_index (int): The stream index. Always 0 for WAV files.
    """

    def __init__(
        self,
        source: str | Path | io.RawIOBase | io.BufferedReader | bytes | torch.Tensor,
    ):
        torch._C._log_api_usage_once("torchcodec.decoders.WavDecoder")

        self._decoder = create_wav_decoder(source)
        self._source = source
        self.stream_index = 0

        metadata_json = _core.get_wav_metadata_from_decoder(self._decoder)
        metadata_dict = json.loads(metadata_json)

        self.metadata = AudioStreamMetadata(
            sample_rate=metadata_dict["sampleRate"],
            num_channels=metadata_dict["numChannels"],
            sample_format=metadata_dict["sampleFormat"],
            duration_seconds=metadata_dict["durationSeconds"],
            stream_index=metadata_dict["streamIndex"],
            codec=metadata_dict["codec"],
            bit_rate=metadata_dict["bitRate"],
            duration_seconds_from_header=metadata_dict["durationSecondsFromHeader"],
            begin_stream_seconds=metadata_dict["beginStreamSeconds"],
            begin_stream_seconds_from_header=None,  # WAV format lacks stream start time metadata
        )

    def get_all_samples(self) -> AudioSamples:
        """Returns all the audio samples from the source.

        To decode samples in a specific range, use
        :meth:`~torchcodec.decoders.WavDecoder.get_samples_played_in_range`.

        Returns:
            AudioSamples: The samples within the file.
        """
        return self.get_samples_played_in_range()

    def get_samples_played_in_range(
        self, start_seconds: float = 0.0, stop_seconds: float | None = None
    ) -> AudioSamples:
        """Returns audio samples in the given range.

        Samples are in the half open range [start_seconds, stop_seconds).

        To decode all the samples from beginning to end, you can call this
        method while leaving ``start_seconds`` and ``stop_seconds`` to their
        default values, or use
        :meth:`~torchcodec.decoders.WavDecoder.get_all_samples` as a more
        convenient alias.

        Args:
            start_seconds (float): Time, in seconds, of the start of the
                range. Default: 0.
            stop_seconds (float or None): Time, in seconds, of the end of the
                range. As a half open range, the end is excluded. Default: None,
                which decodes samples until the end.

        Returns:
            AudioSamples: The samples within the specified range.
        """
        if stop_seconds is not None and not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. It must be less than or equal to stop seconds ({stop_seconds})."
            )

        actual_start_seconds = max(0.0, start_seconds)
        samples, actual_pts = _core.get_wav_samples_in_range(
            self._decoder, actual_start_seconds, stop_seconds
        )
        actual_pts = actual_pts.item()
        assert self.metadata.sample_rate is not None  # make mypy happy
        duration_seconds = samples.shape[1] / self.metadata.sample_rate
        return AudioSamples(
            data=samples,
            pts_seconds=actual_pts,
            duration_seconds=duration_seconds,
            sample_rate=self.metadata.sample_rate,
        )
