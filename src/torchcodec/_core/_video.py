# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn, Tensor

from torchcodec._core._metadata import VideoStreamMetadata, get_container_metadata
from torchcodec._core.ops import (
    add_video_stream,
    create_from_bytes,
    create_from_file,
    create_from_file_like,
    create_from_tensor,
)
from torchcodec.transforms._decoder_transforms import _make_transform_specs

_ERROR_REPORTING_INSTRUCTIONS = """
This should never happen. Please report an issue following the steps in
https://github.com/pytorch/torchcodec/issues/new?assignees=&labels=&projects=&template=bug-report.yml.
"""


def create_video_decoder(
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    *,
    stream_index: int | None = None,
    seek_mode: str = "exact",
    dimension_order: str = "NCHW",
    num_ffmpeg_threads: int = 1,
    device: str = "cpu",
    device_variant: str = "ffmpeg",
    transforms: Sequence | None = None,
    custom_frame_mappings: str | bytes | io.RawIOBase | io.BufferedReader | None = None,
) -> tuple[Tensor, int, VideoStreamMetadata, float, float, int]:

    # Validate seek_mode
    allowed_seek_modes = ("exact", "approximate")
    if seek_mode not in allowed_seek_modes:
        raise ValueError(
            f"Invalid seek mode ({seek_mode}). "
            f"Supported values are {', '.join(allowed_seek_modes)}."
        )

    # Validate seek_mode and custom_frame_mappings compatibility
    if custom_frame_mappings is not None and seek_mode == "approximate":
        raise ValueError(
            "custom_frame_mappings is incompatible with seek_mode='approximate'. "
            "Use seek_mode='custom_frame_mappings' or leave it unspecified to automatically use custom frame mappings."
        )

    # Auto-select custom_frame_mappings seek_mode and process data when mappings are provided
    custom_frame_mappings_data = None
    if custom_frame_mappings is not None:
        seek_mode = "custom_frame_mappings"
        custom_frame_mappings_data = _read_custom_frame_mappings(
            custom_frame_mappings
        )

    # Source-type dispatch
    if isinstance(source, str):
        decoder = create_from_file(source, seek_mode)
    elif isinstance(source, Path):
        decoder = create_from_file(str(source), seek_mode)
    elif isinstance(source, io.RawIOBase) or isinstance(source, io.BufferedReader):
        decoder = create_from_file_like(source, seek_mode)
    elif isinstance(source, bytes):
        decoder = create_from_bytes(source, seek_mode)
    elif isinstance(source, Tensor):
        decoder = create_from_tensor(source, seek_mode)
    elif isinstance(source, io.TextIOBase):
        raise TypeError(
            "source is for reading text, likely from open(..., 'r'). Try with 'rb' for binary reading?"
        )
    elif hasattr(source, "read") and hasattr(source, "seek"):
        decoder = create_from_file_like(source, seek_mode)
    else:
        raise TypeError(
            f"Unknown source type: {type(source)}. "
            "Supported types are str, Path, bytes, Tensor and file-like objects with "
            "read(self, size: int) -> bytes and "
            "seek(self, offset: int, whence: int) -> int methods."
        )

    # Get container metadata and resolve stream_index
    container_metadata = get_container_metadata(decoder)

    if stream_index is None:
        stream_index = container_metadata.best_video_stream_index
    if stream_index is None:
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
    begin_stream_seconds = metadata.begin_stream_seconds

    if metadata.end_stream_seconds is None:
        raise ValueError(
            "The maximum pts value in seconds is unknown. "
            + _ERROR_REPORTING_INSTRUCTIONS
        )
    end_stream_seconds = metadata.end_stream_seconds

    if metadata.num_frames is None:
        raise ValueError(
            "The number of frames is unknown. " + _ERROR_REPORTING_INSTRUCTIONS
        )
    num_frames = metadata.num_frames

    # Validate dimension_order
    allowed_dimension_orders = ("NCHW", "NHWC")
    if dimension_order not in allowed_dimension_orders:
        raise ValueError(
            f"Invalid dimension order ({dimension_order}). "
            f"Supported values are {', '.join(allowed_dimension_orders)}."
        )

    # Validate num_ffmpeg_threads
    if num_ffmpeg_threads is None:
        raise ValueError(f"{num_ffmpeg_threads = } should be an int.")

    # Compute transform specs
    transform_specs = _make_transform_specs(
        transforms,
        input_dims=(metadata.height, metadata.width),
    )

    # Add video stream
    add_video_stream(
        decoder,
        stream_index=stream_index,
        dimension_order=dimension_order,
        num_threads=num_ffmpeg_threads,
        device=device,
        device_variant=device_variant,
        transform_specs=transform_specs,
        custom_frame_mappings=custom_frame_mappings_data,
    )

    return (
        decoder,
        stream_index,
        metadata,
        begin_stream_seconds,
        end_stream_seconds,
        num_frames,
    )


def _read_custom_frame_mappings(
    custom_frame_mappings: str | bytes | io.RawIOBase | io.BufferedReader,
) -> tuple[Tensor, Tensor, Tensor]:
    """Parse custom frame mappings from JSON data and extract frame metadata.

    Args:
        custom_frame_mappings: JSON data containing frame metadata, provided as:
            - A JSON string (str, bytes)
            - A file-like object with a read() method

    Returns:
        A tuple of three tensors:
        - all_frames (Tensor): Presentation timestamps (PTS) for each frame
        - is_key_frame (Tensor): Boolean tensor indicating which frames are key frames
        - duration (Tensor): Duration of each frame
    """
    try:
        input_data = (
            json.load(custom_frame_mappings)
            if hasattr(custom_frame_mappings, "read")
            else json.loads(custom_frame_mappings)
        )
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid custom frame mappings: {e}. It should be a valid JSON string or a file-like object."
        ) from e

    if not input_data or "frames" not in input_data:
        raise ValueError(
            "Invalid custom frame mappings. The input is empty or missing the required 'frames' key."
        )

    first_frame = input_data["frames"][0]
    pts_key = next((key for key in ("pts", "pkt_pts") if key in first_frame), None)
    duration_key = next(
        (key for key in ("duration", "pkt_duration") if key in first_frame), None
    )
    key_frame_present = "key_frame" in first_frame

    if not pts_key or not duration_key or not key_frame_present:
        raise ValueError(
            "Invalid custom frame mappings. The 'pts'/'pkt_pts', 'duration'/'pkt_duration', and 'key_frame' keys are required in the frame metadata."
        )

    all_frames = torch.tensor(
        [int(frame[pts_key]) for frame in input_data["frames"]], dtype=torch.int64
    )
    is_key_frame = torch.tensor(
        [int(frame["key_frame"]) for frame in input_data["frames"]], dtype=torch.bool
    )
    duration = torch.tensor(
        [int(frame[duration_key]) for frame in input_data["frames"]], dtype=torch.int64
    )
    if not (len(all_frames) == len(is_key_frame) == len(duration)):
        raise ValueError("Mismatched lengths in frame index data")
    return all_frames, is_key_frame, duration
