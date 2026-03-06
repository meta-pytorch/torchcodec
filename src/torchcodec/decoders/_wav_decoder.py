# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
from pathlib import Path

import torch
from torchcodec import _core

from torchcodec._core._metadata import get_wav_metadata


def _is_uncompressed_wav(source) -> dict | None:
    """Check if source is an uncompressed WAV file compatible with native decoder.

    Returns parsed metadata dict if compatible, None otherwise.
    """
    try:
        if isinstance(source, str):
            metadata_json = _core._get_wav_metadata_from_file(source)
        elif isinstance(source, Path):
            metadata_json = _core._get_wav_metadata_from_file(str(source))
        else:
            return None

        if not metadata_json:
            return None

        return json.loads(metadata_json)
    except Exception:
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

        if _is_uncompressed_wav(source) is None:
            raise ValueError(
                "Source is not a supported uncompressed WAV file. "
                "For compressed audio formats or non-WAV files, use AudioDecoder instead."
            )

        self._source = source
        self.stream_index = 0
        self.metadata = get_wav_metadata(source)
