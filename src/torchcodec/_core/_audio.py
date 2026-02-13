# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
from pathlib import Path

from torch import Tensor

from torchcodec._core._metadata import AudioStreamMetadata, get_container_metadata
from torchcodec._core.ops import (
    add_audio_stream,
    create_from_bytes,
    create_from_file,
    create_from_file_like,
    create_from_tensor,
)

_ERROR_REPORTING_INSTRUCTIONS = """
This should never happen. Please report an issue following the steps in
https://github.com/pytorch/torchcodec/issues/new?assignees=&labels=&projects=&template=bug-report.yml.
"""


def create_audio_decoder(
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    *,
    stream_index: int | None = None,
    sample_rate: int | None = None,
    num_channels: int | None = None,
) -> tuple[Tensor, int, AudioStreamMetadata]:
    # Source-type dispatch (seek_mode is always "approximate" for audio)
    if isinstance(source, str):
        decoder = create_from_file(source, "approximate")
    elif isinstance(source, Path):
        decoder = create_from_file(str(source), "approximate")
    elif isinstance(source, io.RawIOBase) or isinstance(source, io.BufferedReader):
        decoder = create_from_file_like(source, "approximate")
    elif isinstance(source, bytes):
        decoder = create_from_bytes(source, "approximate")
    elif isinstance(source, Tensor):
        decoder = create_from_tensor(source, "approximate")
    elif isinstance(source, io.TextIOBase):
        raise TypeError(
            "source is for reading text, likely from open(..., 'r'). Try with 'rb' for binary reading?"
        )
    elif hasattr(source, "read") and hasattr(source, "seek"):
        decoder = create_from_file_like(source, "approximate")
    else:
        raise TypeError(
            f"Unknown source type: {type(source)}. "
            "Supported types are str, Path, bytes, Tensor and file-like objects with "
            "read(self, size: int) -> bytes and "
            "seek(self, offset: int, whence: int) -> int methods."
        )

    container_metadata = get_container_metadata(decoder)

    if stream_index is None:
        stream_index = container_metadata.best_audio_stream_index
    if stream_index is None:
        raise ValueError(
            "The best audio stream is unknown and there is no specified stream. "
            + _ERROR_REPORTING_INSTRUCTIONS
        )
    if stream_index >= len(container_metadata.streams):
        raise ValueError(
            f"The stream at index {stream_index} is not a valid stream."
        )

    metadata = container_metadata.streams[stream_index]
    if not isinstance(metadata, AudioStreamMetadata):
        raise ValueError(
            f"The stream at index {stream_index} is not an audio stream. "
        )

    add_audio_stream(
        decoder,
        stream_index=stream_index,
        sample_rate=sample_rate,
        num_channels=num_channels,
    )

    return decoder, stream_index, metadata
