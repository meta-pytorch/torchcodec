from typing import Any

from torch import Tensor

from torchcodec import _core


# TODO MultiStreamEncoder: the stream_index values here are per media-type,
# while everywhere else in the code base (and particularly in the public decoder
# APIs) they are absolute across all media types. That'll quickly becomes
# confusing, and we should definitely not expose this one as-is. We should either:
# - keep it private but rename it to something that's not stream_index
# - make it absolute per container, if we ever want to expose it.
class _VideoStream:
    def __init__(self, encoder_tensor: Tensor, stream_index: int):
        self._encoder_tensor = encoder_tensor
        self._stream_index = stream_index

    def write(self, frames: Tensor) -> None:
        _core.streaming_encoder_add_frames(
            self._encoder_tensor, frames, self._stream_index
        )


class _AudioStream:
    def __init__(self, encoder_tensor: Tensor, stream_index: int):
        self._encoder_tensor = encoder_tensor
        self._stream_index = stream_index

    def write(self, samples: Tensor) -> None:
        _core.streaming_encoder_add_samples(
            self._encoder_tensor, samples, self._stream_index
        )


class StreamingEncoder:
    def __init__(self):
        self._encoder_tensor = _core.create_streaming_encoder()

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
        preset = str(preset) if isinstance(preset, int) else preset
        stream_index = _core.streaming_encoder_add_video_stream(
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
        return _VideoStream(self._encoder_tensor, stream_index)

    def add_audio(
        self,
        *,
        sample_rate: int,
        num_channels: int,
        bit_rate: int | None = None,
        # TODO MultiStreamEncoder: Decide on public API for 'output' params
        output_num_channels: int | None = None,
        output_sample_rate: int | None = None,
    ) -> _AudioStream:
        stream_index = _core.streaming_encoder_add_audio_stream(
            self._encoder_tensor,
            sample_rate=sample_rate,
            num_channels=num_channels,
            bit_rate=bit_rate,
            output_num_channels=output_num_channels,
            output_sample_rate=output_sample_rate,
        )
        return _AudioStream(self._encoder_tensor, stream_index)

    # TODO MultiStreamEncoder: Maybe there should 2 separate methods, one for
    # file, one for file-like.
    def open(self, dest, *, format: str | None = None) -> "StreamingEncoder":
        if format is not None:
            _core.streaming_encoder_open_file_like(self._encoder_tensor, format, dest)
        else:
            _core.streaming_encoder_open_file(self._encoder_tensor, str(dest))
        return self

    def close(self) -> None:
        _core.streaming_encoder_close(self._encoder_tensor)

    def __enter__(self) -> "StreamingEncoder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
