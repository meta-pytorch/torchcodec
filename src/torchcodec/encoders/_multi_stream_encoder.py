from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from torchcodec import _core


class _VideoStream:
    def __init__(self, encoder_tensor: Tensor):
        self._encoder_tensor = encoder_tensor

    def write(self, frames: Tensor) -> None:
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4:
            raise ValueError(f"Expected 3D or 4D frames, got {frames.shape = }.")
        if frames.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 frames, got {frames.dtype = }.")
        _core.streaming_encoder_add_frames(self._encoder_tensor, frames)


class _AudioStream:
    def __init__(self, encoder_tensor: Tensor):
        self._encoder_tensor = encoder_tensor

    def write(self, samples: Tensor) -> None:
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        if samples.ndim != 2:
            raise ValueError(f"Expected 1D or 2D samples, got {samples.shape = }.")
        if samples.dtype != torch.float32:
            raise ValueError(f"Expected float32 samples, got {samples.dtype = }.")
        _core.streaming_encoder_add_samples(self._encoder_tensor, samples)


class StreamingEncoder:
    def __init__(self, dest: str | Path):
        self._encoder_tensor = _core.create_streaming_encoder_to_file(str(dest))
        self._opened = False
        self._closed = False

    def add_video(
        self,
        *,
        height: int,
        width: int,
        frame_rate: float,
        device: str = "cpu",
        codec: str | None = None,
        pixel_format: str | None = None,
        crf: int | float | None = None,
        preset: str | int | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> _VideoStream:
        if self._opened:
            raise RuntimeError("Cannot add streams after open() has been called.")
        preset = str(preset) if isinstance(preset, int) else preset
        _core.streaming_encoder_add_video_stream(
            self._encoder_tensor,
            height=height,
            width=width,
            frame_rate=frame_rate,
            device=device,
            codec=codec,
            pixel_format=pixel_format,
            crf=crf,
            preset=preset,
            extra_options=[
                str(x) for k, v in (extra_options or {}).items() for x in (k, v)
            ],
        )
        return _VideoStream(self._encoder_tensor)

    def add_audio(
        self,
        *,
        sample_rate: int,
        num_channels: int,
        bit_rate: int | None = None,
    ) -> _AudioStream:
        if self._opened:
            raise RuntimeError("Cannot add streams after open() has been called.")
        _core.streaming_encoder_add_audio_stream(
            self._encoder_tensor,
            sample_rate=sample_rate,
            num_channels=num_channels,
            bit_rate=bit_rate,
        )
        return _AudioStream(self._encoder_tensor)

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("Encoder is already open.")
        self._opened = True
        _core.streaming_encoder_open(self._encoder_tensor)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _core.streaming_encoder_close(self._encoder_tensor)
