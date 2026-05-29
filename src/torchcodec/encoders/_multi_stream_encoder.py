from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from torchcodec import _core


# TODO MultiStreamEncoder: the stream_index values here are per media-type,
# while everywhere else in the code base (and particularly in the public decoder
# APIs) they are absolute across all media types. That'll quickly becomes
# confusing, and we should definitely not expose this one as-is. We should either:
# - keep it private but rename it to something that's not stream_index
# - make it absolute per container, if we ever want to expose it.
class VideoStream:
    """A video stream within an :class:`Encoder`.

    Returned by :meth:`Encoder.add_video`. Use :meth:`add_frames` to feed
    video frames into this stream.
    """

    def __init__(self, encoder_tensor: Tensor, stream_index: int):
        self._encoder_tensor = encoder_tensor
        self._stream_index = stream_index

    def add_frames(self, frames: Tensor) -> None:
        """Add video frames to this stream.

        Args:
            frames (``torch.Tensor``): The frames to encode. This must be a 4D
                tensor of shape ``(N, C, H, W)`` where N is the number of
                frames, C is 3 channels (RGB), H is height, and W is width.
                Values must be uint8 in the range ``[0, 255]``. The device of
                the tensor must match the ``device`` passed to
                :meth:`Encoder.add_video`.
        """
        _core.streaming_encoder_add_frames(
            self._encoder_tensor, frames, self._stream_index
        )


class AudioStream:
    """An audio stream within an :class:`Encoder`.

    Returned by :meth:`Encoder.add_audio`. Use :meth:`add_samples` to feed
    audio samples into this stream.
    """

    def __init__(self, encoder_tensor: Tensor, stream_index: int):
        self._encoder_tensor = encoder_tensor
        self._stream_index = stream_index

    def add_samples(self, samples: Tensor) -> None:
        """Add audio samples to this stream.

        Args:
            samples (``torch.Tensor``): The samples to encode. This must be a
                2D tensor of shape ``(num_channels, num_samples)``. Values must
                be float values in ``[-1, 1]``. The number of channels must
                match the ``num_channels`` passed to :meth:`Encoder.add_audio`.
        """
        _core.streaming_encoder_add_samples(
            self._encoder_tensor, samples, self._stream_index
        )


class Encoder:
    """A multi-stream encoder for encoding video and/or audio streams.

    Unlike :class:`VideoEncoder` and :class:`AudioEncoder` which encode a
    single stream in one shot, ``Encoder`` supports multiple streams and
    incremental (streaming) encoding. Frames and samples can be added
    progressively, which is useful when data is generated on-the-fly or when
    encoding both audio and video into the same container.

    Use :meth:`add_video` and :meth:`add_audio` to configure output streams,
    then open an output destination with :meth:`open_file` or
    :meth:`open_file_like`, feed data via the returned stream objects, and
    finally call :meth:`close` (or use the encoder as a context manager).

    Example:

        .. code-block:: python

            encoder = Encoder()
            video_stream = encoder.add_video(height=256, width=256, frame_rate=30)
            audio_stream = encoder.add_audio(sample_rate=16000, num_channels=1)
            with encoder.open_file("output.mp4"):
                video_stream.add_frames(frames_tensor)
                audio_stream.add_samples(samples_tensor)
                # Add more frames by calling video_stream.add_frames again
                # Add more samples by calling audio_stream.add_samples again

        To encode to a file-like object (e.g. ``io.BytesIO()``), use
        :meth:`open_file_like` instead:

        .. code-block:: python

            import io

            buf = io.BytesIO()
            encoder = Encoder()
            video_stream = encoder.add_video(height=256, width=256, frame_rate=30)
            with encoder.open_file_like(buf, format="mp4"):
                video_stream.add_frames(frames_tensor)
            encoded_bytes = buf.getvalue()
            # Optionally convert to a uint8 tensor of bytes with
            # bytes_tensor = torch.frombuffer(encoded_bytes, dtype=torch.uint8)
    """

    def __init__(self):
        self._encoder_tensor = _core.create_streaming_encoder()

    def add_video(
        self,
        *,
        height: int,
        width: int,
        frame_rate: float,
        device: str | torch.device | None = None,
        codec: str | None = None,
        pixel_format: str | None = None,
        crf: int | float | None = None,
        preset: str | int | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> VideoStream:
        """Add a video stream to the encoder.

        Must be called before :meth:`open_file` or :meth:`open_file_like`.

        Args:
            height (int): The height of the **input** video frames.
            width (int): The width of the **input** video frames.
            frame_rate (float): The frame rate of the **input** video frames.
                Also defines the encoded **output** frame rate.
            device (str or torch.device, optional): The device to use for
                encoding, e.g.  ``"cpu"`` or ``"cuda"``. If ``None`` (default), uses
                the current default device.
            codec (str, optional): The codec to use for encoding (e.g.,
                ``"libx264"``). If not specified, the default codec for the
                container format will be used.
                See :ref:`codec_selection` for details.
            pixel_format (str, optional): The pixel format for encoding (e.g.,
                ``"yuv420p"``). If not specified, uses codec's default format.
                Must be left as ``None`` when encoding on CUDA.
                See :ref:`pixel_format` for details.
            crf (int or float, optional): Constant Rate Factor for encoding
                quality. Lower values mean better quality. Valid range depends
                on the encoder (e.g. 0-51 for libx264). Defaults to None (which
                will use encoder's default). See :ref:`crf` for details.
            preset (str or int, optional): Encoder option that controls the
                tradeoff between encoding speed and compression (output size).
                Commonly a string: ``"fast"``, ``"medium"``, ``"slow"``.
                Defaults to None (which will use encoder's default).
                See :ref:`preset` for details.
            extra_options (dict[str, Any], optional): A dictionary of additional
                encoder options to pass, e.g. ``{"qp": 5, "tune": "film"}``.
                See :ref:`extra_options` for details.

        Returns:
            A video stream object. Use its :meth:`~VideoStream.add_frames`
            method to feed frames into the stream.
        """
        if device is None:
            device = torch.get_default_device()
        device = str(device)
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
        return VideoStream(self._encoder_tensor, stream_index)

    def add_audio(
        self,
        *,
        sample_rate: int,
        num_channels: int,
        bit_rate: int | None = None,
        out_num_channels: int | None = None,
        out_sample_rate: int | None = None,
    ) -> AudioStream:
        """Add an audio stream to the encoder.

        Must be called before :meth:`open_file` or :meth:`open_file_like`.

        Args:
            sample_rate (int): The sample rate of the **input** samples.
            num_channels (int): The number of channels of the **input** samples.
            bit_rate (int, optional): The output bit rate. Encoders typically
                support a finite set of bit rate values, so ``bit_rate`` will be
                matched to one of those supported values. The default is chosen
                by FFmpeg.
            out_num_channels (int, optional): The number of channels of the
                encoded output. By default, the input ``num_channels`` is used.
            out_sample_rate (int, optional): The sample rate of the encoded
                output. By default, the input ``sample_rate`` is used.

        Returns:
            An audio stream object. Use its :meth:`~AudioStream.add_samples`
            method to feed samples into the stream.
        """
        stream_index = _core.streaming_encoder_add_audio_stream(
            self._encoder_tensor,
            sample_rate=sample_rate,
            num_channels=num_channels,
            bit_rate=bit_rate,
            output_num_channels=out_num_channels,
            output_sample_rate=out_sample_rate,
        )
        return AudioStream(self._encoder_tensor, stream_index)

    def open_file(self, dest: str | Path) -> "Encoder":
        """Open a file for writing the encoded output.

        Must be called after all streams have been added via :meth:`add_video`
        and/or :meth:`add_audio`. The file extension determines the container
        format (e.g. ``.mp4``, ``.mkv``).

        Args:
            dest (str or ``pathlib.Path``): The path to the output file.

        Returns:
            Encoder: Returns ``self`` for method chaining.
        """
        _core.streaming_encoder_open_file(self._encoder_tensor, str(dest))
        return self

    def open_file_like(self, dest, *, format: str) -> "Encoder":
        """Open a file-like object for writing the encoded output.

        Must be called after all streams have been added via :meth:`add_video`
        and/or :meth:`add_audio`.

        Args:
            dest: A file-like object that supports ``write()`` and ``seek()``
                methods, such as ``io.BytesIO()``, an open file in binary write
                mode, etc. Methods must have the following signature:
                ``write(data: bytes) -> int`` and ``seek(offset: int, whence:
                int = 0) -> int``.
            format (str): The container format of the encoded output, e.g.
                ``"mp4"``, ``"mov"``, ``"mkv"``, ``"avi"``, ``"webm"``, etc.

        Returns:
            Encoder: Returns ``self`` for method chaining.
        """
        _core.streaming_encoder_open_file_like(self._encoder_tensor, format, dest)
        return self

    def close(self) -> None:
        """Flush all remaining data and close the encoder.

        This must be called when encoding is complete to ensure all buffered
        data is written. Using the encoder as a context manager (``with``
        statement) calls this automatically.
        """
        _core.streaming_encoder_close(self._encoder_tensor)

    def __enter__(self) -> "Encoder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
