# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
from pathlib import Path

import torch
from torchcodec import _core, AudioSamples
from torchcodec._core._decoder_utils import create_wav_decoder
from torchcodec._core._metadata import AudioStreamMetadata


class WavDecoder:
    # TODO: Docstrings
    def __init__(self, source: str | Path):
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
        return self.get_samples_played_in_range()

    def get_samples_played_in_range(
        self, start_seconds: float = 0.0, stop_seconds: float | None = None
    ) -> AudioSamples:
        if stop_seconds is not None and not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. It must be less than or equal to stop seconds ({stop_seconds})."
            )

        actual_start_seconds = max(0.0, start_seconds)
        frames, actual_pts = _core.get_wav_samples_in_range(
            self._decoder, actual_start_seconds, stop_seconds
        )
        actual_pts = actual_pts.item()
        assert self.metadata.sample_rate is not None  # make mypy happy
        duration_seconds = frames.shape[1] / self.metadata.sample_rate
        return AudioSamples(
            data=frames,
            pts_seconds=actual_pts,
            duration_seconds=duration_seconds,
            sample_rate=self.metadata.sample_rate,
        )
