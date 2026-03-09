# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
from pathlib import Path

import torch
from torchcodec import _core

from torchcodec._core._metadata import _create_wav_metadata_from_dict


def _try_get_wav_metadata(source) -> dict | None:
    """Try to extract WAV metadata if source is a supported uncompressed WAV.

    Returns parsed metadata dict if compatible, None otherwise.
    """
    try:
        if isinstance(source, str):
            metadata_json = _core._get_wav_metadata_from_file(source)
        elif isinstance(source, Path):
            metadata_json = _core._get_wav_metadata_from_file(str(source))
        else:
            return None

        return json.loads(metadata_json)
    except Exception as e:
        print(f"Error occurred while processing WAV file: {e}")
        return None


class WavDecoder:
    """A validator for uncompressed WAV audio files.

    This class validates WAV file format and extracts metadata.
    For compressed audio formats or non-WAV files, use AudioDecoder instead.

    Args:
        source (str or ``Pathlib.path``): Path to a WAV file.

    Attributes:
        metadata (AudioStreamMetadata): Metadata of the audio stream.
        stream_index (int): Always 0 for WAV files (single stream).

    Raises:
        ValueError: If the source is not a supported uncompressed WAV file.
    """

    def __init__(self, source: str | Path):
        torch._C._log_api_usage_once("torchcodec.decoders.WavDecoder")

        metadata_dict = _try_get_wav_metadata(source)
        if metadata_dict is None:
            raise ValueError(
                "Source is not a supported uncompressed WAV file. "
                "For compressed audio formats or non-WAV files, use AudioDecoder instead."
            )

        self._source = source
        self.stream_index = 0
        self.metadata = _create_wav_metadata_from_dict(metadata_dict)
