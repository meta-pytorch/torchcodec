# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import contextvars
import io
import json

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor
from torchcodec import _core as core

ERROR_REPORTING_INSTRUCTIONS = """
This should never happen. Please report an issue following the steps in
https://github.com/pytorch/torchcodec/issues/new?assignees=&labels=&projects=&template=bug-report.yml.
"""


def create_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
) -> Tensor:
    if isinstance(source, str):
        return core.create_from_file(source, seek_mode)
    elif isinstance(source, Path):
        return core.create_from_file(str(source), seek_mode)
    elif isinstance(source, io.RawIOBase) or isinstance(source, io.BufferedReader):
        return core.create_from_file_like(source, seek_mode)
    elif isinstance(source, bytes):
        return core.create_from_bytes(source, seek_mode)
    elif isinstance(source, Tensor):
        return core.create_from_tensor(source, seek_mode)
    elif isinstance(source, io.TextIOBase):
        raise TypeError(
            "source is for reading text, likely from open(..., 'r'). Try with 'rb' for binary reading?"
        )
    elif hasattr(source, "read") and hasattr(source, "seek"):
        # This check must be after checking for text-based reading. Also placing
        # it last in general to be defensive: hasattr is a blunt instrument. We
        # could use the inspect module to check for methods with the right
        # signature.
        return core.create_from_file_like(source, seek_mode)

    raise TypeError(
        f"Unknown source type: {type(source)}. "
        "Supported types are str, Path, bytes, Tensor and file-like objects with "
        "read(self, size: int) -> bytes and "
        "seek(self, offset: int, whence: int) -> int methods."
    )


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


def _is_uncompressed_wav(
    source,
    stream_index: int | None = None,
    sample_rate: int | None = None,
    num_channels: int | None = None,
) -> dict | None:
    """Check if source is an uncompressed WAV file compatible with native decoder.

    Returns metadata dict if compatible, None otherwise (not WAV, unsupported format,
    or requested parameters don't match the source).
    """
    try:
        if isinstance(source, str):
            metadata_json = core.get_wav_metadata_from_file(
                source, stream_index, sample_rate, num_channels
            )
        elif isinstance(source, Path):
            metadata_json = core.get_wav_metadata_from_file(
                str(source), stream_index, sample_rate, num_channels
            )
        elif isinstance(source, bytes):
            buffer = torch.frombuffer(source, dtype=torch.uint8)
            metadata_json = core.get_wav_metadata_from_tensor(
                buffer, stream_index, sample_rate, num_channels
            )
        elif isinstance(source, Tensor):
            metadata_json = core.get_wav_metadata_from_tensor(
                source, stream_index, sample_rate, num_channels
            )
        else:
            # File-like objects - read all data to get full metadata
            current_pos = source.seek(0, io.SEEK_CUR)
            source.seek(0)
            data = source.read()
            source.seek(current_pos)
            if len(data) < 12:
                return None
            buffer = torch.frombuffer(data, dtype=torch.uint8)
            metadata_json = core.get_wav_metadata_from_tensor(
                buffer, stream_index, sample_rate, num_channels
            )

        if not metadata_json:
            return None

        return json.loads(metadata_json)
    except Exception:
        # In the case of an error, fall back to FFmpeg decoder
        return None


def decode_wav(
    source: str | Path | bytes | Tensor | io.RawIOBase | io.BufferedReader,
    start_seconds: float = 0.0,
    stop_seconds: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Decode audio samples from a WAV file using the native decoder.

    Args:
        source: The WAV audio source - can be a file path (str or Path),
            raw bytes, a uint8 tensor containing WAV data, or a file-like object.
        start_seconds: Start time in seconds for the audio range.
        stop_seconds: Stop time in seconds (exclusive). None means decode to end.

    Returns:
        A tuple of (samples, pts_seconds) where:
        - samples: Float32 tensor of shape (num_channels, num_samples) normalized to [-1, 1]
        - pts_seconds: Float64 tensor containing the PTS of the first sample

    Raises:
        RuntimeError: If the WAV format is not supported (compressed formats).
    """
    import warnings

    if isinstance(source, str):
        return core.decode_wav_from_file(source, start_seconds, stop_seconds)
    elif isinstance(source, Path):
        return core.decode_wav_from_file(str(source), start_seconds, stop_seconds)
    elif isinstance(source, bytes):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            buffer = torch.frombuffer(source, dtype=torch.uint8)
        return core.decode_wav_from_tensor(buffer, start_seconds, stop_seconds)
    elif isinstance(source, Tensor):
        return core.decode_wav_from_tensor(source, start_seconds, stop_seconds)
    elif hasattr(source, "read") and hasattr(source, "seek"):
        # File-like object - read all data and pass to tensor version
        source.seek(0)
        data = source.read()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            buffer = torch.frombuffer(data, dtype=torch.uint8)
        return core.decode_wav_from_tensor(buffer, start_seconds, stop_seconds)
    else:
        raise TypeError(
            f"Unsupported source type: {type(source)}. "
            "Expected str, Path, bytes, Tensor, or file-like object."
        )
