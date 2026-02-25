# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import contextvars
import io
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import device as torch_device, nn, Tensor

from torchcodec._core._metadata import (
    AudioStreamMetadata,
    get_container_metadata,
    VideoStreamMetadata,
)
from torchcodec._core.ops import (
    add_audio_stream,
    add_video_stream,
    create_from_bytes,
    create_from_file,
    create_from_file_like,
    create_from_tensor,
)
from torchcodec.transforms import DecoderTransform
from torchcodec.transforms._decoder_transforms import _make_transform_specs


_ERROR_REPORTING_INSTRUCTIONS = """
This should never happen. Please report an issue following the steps in
https://github.com/pytorch/torchcodec/issues/new?assignees=&labels=&projects=&template=bug-report.yml.
"""


# Thread-local and async-safe storage for the current CUDA backend
_CUDA_BACKEND: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_CUDA_BACKEND", default="ffmpeg"
)


@contextmanager
def set_cuda_backend(backend: str) -> Generator[None, None, None]:
    """Context Manager to set the CUDA backend for :class:`~torchcodec.decoders.VideoDecoder`.

    This context manager allows you to specify which CUDA backend implementation
    to use when creating :class:`~torchcodec.decoders.VideoDecoder` instances
    with CUDA devices.

    .. note::
        **We recommend trying the "beta" backend instead of the default "ffmpeg"
        backend!** The beta backend is faster, and will eventually become the
        default in future versions. It may have rough edges that we'll polish
        over time, but it's already quite stable and ready for adoption. Let us
        know what you think!

    Only the creation of the decoder needs to be inside the context manager, the
    decoding methods can be called outside of it. You still need to pass
    ``device="cuda"`` when creating the
    :class:`~torchcodec.decoders.VideoDecoder` instance. If a CUDA device isn't
    specified, this context manager will have no effect. See example below.

    This is thread-safe and async-safe.

    Args:
        backend (str): The CUDA backend to use. Can be "ffmpeg" (default) or
            "beta". We recommend trying "beta" as it's faster!

    Example:
        >>> with set_cuda_backend("beta"):
        ...     decoder = VideoDecoder("video.mp4", device="cuda")
        ...
        ... # Only the decoder creation needs to be part of the context manager.
        ... # Decoder will now the beta CUDA implementation:
        ... decoder.get_frame_at(0)
    """
    backend = backend.lower()
    if backend not in ("ffmpeg", "beta"):
        raise ValueError(
            f"Invalid CUDA backend ({backend}). Supported values are 'ffmpeg' and 'beta'."
        )

    previous_state = _CUDA_BACKEND.set(backend)
    try:
        yield
    finally:
        _CUDA_BACKEND.reset(previous_state)


def _get_cuda_backend() -> str:
    return _CUDA_BACKEND.get()


def create_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
) -> Tensor:
    if isinstance(source, str):
        return create_from_file(source, seek_mode)
    elif isinstance(source, Path):
        return create_from_file(str(source), seek_mode)
    elif isinstance(source, io.RawIOBase) or isinstance(source, io.BufferedReader):
        return create_from_file_like(source, seek_mode)
    elif isinstance(source, bytes):
        return create_from_bytes(source, seek_mode)
    elif isinstance(source, Tensor):
        return create_from_tensor(source, seek_mode)
    elif isinstance(source, io.TextIOBase):
        raise TypeError(
            "source is for reading text, likely from open(..., 'r'). Try with 'rb' for binary reading?"
        )
    elif hasattr(source, "read") and hasattr(source, "seek"):
        return create_from_file_like(source, seek_mode)

    raise TypeError(
        f"Unknown source type: {type(source)}. "
        "Supported types are str, Path, bytes, Tensor and file-like objects with "
        "read(self, size: int) -> bytes and "
        "seek(self, offset: int, whence: int) -> int methods."
    )


def create_audio_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
    stream_index: int | None = None,
    sample_rate: int | None = None,
    num_channels: int | None = None,
) -> tuple[Tensor, int, AudioStreamMetadata]:

    decoder = create_decoder(source=source, seek_mode=seek_mode)

    container_metadata = get_container_metadata(decoder)

    if stream_index is None:
        stream_index = container_metadata.best_audio_stream_index
        if stream_index is None:
            raise ValueError(
                "The best audio stream is unknown and there is no specified stream. "
                + _ERROR_REPORTING_INSTRUCTIONS
            )

    if stream_index >= len(container_metadata.streams):
        raise ValueError(f"The stream at index {stream_index} is not a valid stream.")

    metadata = container_metadata.streams[stream_index]
    if not isinstance(metadata, AudioStreamMetadata):
        raise ValueError(f"The stream at index {stream_index} is not an audio stream.")

    add_audio_stream(
        decoder,
        stream_index=stream_index,
        sample_rate=sample_rate,
        num_channels=num_channels,
    )

    return (decoder, stream_index, metadata)


def _get_and_validate_stream_metadata(
    *,
    decoder: Tensor,
    stream_index: int | None = None,
) -> tuple[VideoStreamMetadata, int, float, float, int]:
    """Get and validate video stream metadata from a decoder.

    Args:
        decoder: The decoder tensor.
        stream_index: The stream index to use, or None to use the best stream.

    Returns:
        A tuple of (metadata, stream_index, begin_stream_seconds,
        end_stream_seconds, num_frames).
    """
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

    return (
        metadata,
        stream_index,
        begin_stream_seconds,
        end_stream_seconds,
        num_frames,
    )


def create_video_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
    stream_index: int | None = None,
    dimension_order: str = "NCHW",
    num_ffmpeg_threads: int = 1,
    device: str | torch_device | None = None,
    transforms: Sequence[DecoderTransform | nn.Module] | None = None,
    custom_frame_mappings: tuple[Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, VideoStreamMetadata, int, float, float, int, str]:
    """Create a video decoder and add a video stream.

    This function consolidates the creation of a decoder and adding a video stream
    into a single operation, performing all necessary validation.

    Args:
        source: The source of the video.
        seek_mode: The seek mode for the decoder.
        stream_index: The stream index to decode, or None to use the best stream.
        dimension_order: The dimension order for decoded frames.
        num_ffmpeg_threads: Number of FFmpeg threads for CPU decoding.
        device: The device for decoding.
        transforms: Optional sequence of transforms to apply.
        custom_frame_mappings: Optional pre-processed frame mappings data.

    Returns:
        A tuple of (decoder, metadata, stream_index, begin_stream_seconds,
        end_stream_seconds, num_frames, device_variant).
    """
    decoder = create_decoder(source=source, seek_mode=seek_mode)

    (
        metadata,
        stream_index,
        begin_stream_seconds,
        end_stream_seconds,
        num_frames,
    ) = _get_and_validate_stream_metadata(decoder=decoder, stream_index=stream_index)

    if device is None:
        device = str(torch.get_default_device())
    elif isinstance(device, torch_device):
        device = str(device)

    device_variant = _get_cuda_backend()
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

    return (
        decoder,
        metadata,
        stream_index,
        begin_stream_seconds,
        end_stream_seconds,
        num_frames,
        device_variant,
    )
