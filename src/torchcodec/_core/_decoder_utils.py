# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import io
from collections.abc import Sequence
from pathlib import Path

from torch import nn, Tensor
from torchcodec._core._metadata import (
    AudioStreamMetadata,
    get_container_metadata,
    VideoStreamMetadata,
)
from torchcodec._core.ops import (
    add_video_stream,
    create_audio_decoder as _create_audio_decoder_op,
    create_decoder,
    get_active_stream_index,
)
from torchcodec.transforms import DecoderTransform
from torchcodec.transforms._decoder_transforms import _make_transform_specs


_ERROR_REPORTING_INSTRUCTIONS = """
This should never happen. Please report an issue following the steps in
https://github.com/pytorch/torchcodec/issues/new?assignees=&labels=&projects=&template=bug-report.yml.
"""


def create_audio_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
    stream_index: int | None = None,
    sample_rate: int | None = None,
    num_channels: int | None = None,
) -> tuple[Tensor, int, AudioStreamMetadata]:
    # Use unified audio decoder op that creates decoder and adds stream in one step
    # Validation is done inside _create_audio_decoder_op
    decoder = _create_audio_decoder_op(
        source,
        stream_index=stream_index,
        sample_rate=sample_rate,
        num_channels=num_channels,
    )

    # Get the actual stream index that was used (in case stream_index was None)
    actual_stream_index = get_active_stream_index(decoder)

    # Get metadata for the stream
    container_metadata = get_container_metadata(decoder)
    metadata = container_metadata.streams[actual_stream_index]
    if not isinstance(metadata, AudioStreamMetadata):
        raise ValueError(
            f"The stream at index {actual_stream_index} is not an audio stream."
        )

    return (decoder, actual_stream_index, metadata)


def _get_and_validate_video_stream_metadata(
    *,
    decoder: Tensor,
    stream_index: int | None = None,
) -> tuple[VideoStreamMetadata, int]:
    container_metadata = get_container_metadata(decoder)

    if stream_index is None:
        if (stream_index := container_metadata.best_video_stream_index) is None:
            raise ValueError(
                "The best video stream is unknown and there is no specified stream. "
                + _ERROR_REPORTING_INSTRUCTIONS
            )

    if stream_index >= len(container_metadata.streams):
        raise ValueError(f"The stream index {stream_index} is not a valid stream.")

    metadata = container_metadata.streams[stream_index]
    if not isinstance(metadata, VideoStreamMetadata):
        raise ValueError(f"The stream at index {stream_index} is not a video stream. ")

    if metadata.begin_stream_seconds is None:
        raise ValueError(
            "The minimum pts value in seconds is unknown. "
            + _ERROR_REPORTING_INSTRUCTIONS
        )

    if metadata.end_stream_seconds is None:
        raise ValueError(
            "The maximum pts value in seconds is unknown. "
            + _ERROR_REPORTING_INSTRUCTIONS
        )

    if metadata.num_frames is None:
        raise ValueError(
            "The number of frames is unknown. " + _ERROR_REPORTING_INSTRUCTIONS
        )

    return (metadata, stream_index)


def create_video_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
    stream_index: int | None = None,
    dimension_order: str = "NCHW",
    num_ffmpeg_threads: int = 1,
    device: str,
    device_variant: str = "ffmpeg",
    transforms: Sequence[DecoderTransform | nn.Module] | None = None,
    custom_frame_mappings: tuple[Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, int, VideoStreamMetadata]:

    decoder = create_decoder(source=source, seek_mode=seek_mode)

    (
        metadata,
        stream_index,
    ) = _get_and_validate_video_stream_metadata(
        decoder=decoder, stream_index=stream_index
    )

    transform_specs = _make_transform_specs(
        transforms,
        input_dims=(metadata.height, metadata.width),
    )

    add_video_stream(
        decoder,
        stream_index=stream_index,
        dimension_order=dimension_order,
        num_threads=num_ffmpeg_threads,
        device=device,
        device_variant=device_variant,
        transform_specs=transform_specs,
        custom_frame_mappings=custom_frame_mappings,
    )

    return (decoder, stream_index, metadata)
