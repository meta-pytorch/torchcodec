from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from torchcodec import _core


class VideoEncoder:
    """A video encoder.

    Args:
        frames (``torch.Tensor``): The frames to encode. This must be a 4D
            tensor of shape ``(N, C, H, W)`` where N is the number of frames,
            C is 3 channels (RGB), H is height, and W is width.
            Values must be uint8 in the range ``[0, 255]``.
            The device of the frames tensor will be used for encoding.
        frame_rate (float): The frame rate of the **input** ``frames``. Also defines the encoded **output** frame rate.
    """

    def __init__(self, frames: Tensor, *, frame_rate: float):
        torch._C._log_api_usage_once("torchcodec.encoders.VideoEncoder")
        if not isinstance(frames, Tensor):
            raise ValueError(f"Expected frames to be a Tensor, got {type(frames) = }.")
        if frames.ndim != 4:
            raise ValueError(f"Expected 4D frames, got {frames.shape = }.")
        if frames.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 frames, got {frames.dtype = }.")
        if frame_rate <= 0:
            raise ValueError(f"{frame_rate = } must be > 0.")

        self._frames = frames
        self._frame_rate = frame_rate

    def to_file(
        self,
        dest: str | Path,
        *,
        codec: str | None = None,
        pixel_format: str | None = None,
        crf: int | float | None = None,
        preset: str | int | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> None:
        """Encode frames into a file.

        Args:
            dest (str or ``pathlib.Path``): The path to the output file, e.g.
                ``video.mp4``. The extension of the file determines the video
                container format.
            codec (str, optional): The codec to use for encoding (e.g., "libx264",
                "h264"). If not specified, the default codec
                for the container format will be used.
                See :ref:`codec_selection` for details.
            pixel_format (str, optional): The pixel format for encoding (e.g.,
                "yuv420p", "yuv444p"). If not specified, uses codec's default format.
                See :ref:`pixel_format` for details.
            crf (int or float, optional): Constant Rate Factor for encoding quality. Lower values
                mean better quality. Valid range depends on the encoder (e.g.  0-51 for libx264).
                Defaults to None (which will use encoder's default).
                See :ref:`crf` for details.
            preset (str or int, optional): Encoder option that controls the tradeoff between
                encoding encoding speed and compression (output size). Valid on the encoder (commonly
                a string: "fast", "medium", "slow"). Defaults to None
                (which will use encoder's default).
                See :ref:`preset` for details.
            extra_options (dict[str, Any], optional): A dictionary of additional
                encoder options to pass, e.g. ``{"qp": 5, "tune": "film"}``.
                See :ref:`extra_options` for details.
        """
        preset = str(preset) if isinstance(preset, int) else preset
        _core.encode_video_to_file(
            frames=self._frames,
            frame_rate=self._frame_rate,
            filename=str(dest),
            codec=codec,
            pixel_format=pixel_format,
            crf=crf,
            preset=preset,
            extra_options=[
                str(x) for k, v in (extra_options or {}).items() for x in (k, v)
            ],
        )

    def to_tensor(
        self,
        format: str,
        *,
        codec: str | None = None,
        pixel_format: str | None = None,
        crf: int | float | None = None,
        preset: str | int | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> Tensor:
        """Encode frames into raw bytes, as a 1D uint8 Tensor.

        Args:
            format (str): The container format of the encoded frames, e.g. "mp4", "mov",
                    "mkv", "avi", "webm", "flv", etc.
            codec (str, optional): The codec to use for encoding (e.g., "libx264",
                "h264"). If not specified, the default codec
                for the container format will be used.
                See :ref:`codec_selection` for details.
            pixel_format (str, optional): The pixel format to encode frames into (e.g.,
                "yuv420p", "yuv444p"). If not specified, uses codec's default format.
                See :ref:`pixel_format` for details.
            crf (int or float, optional): Constant Rate Factor for encoding quality. Lower values
                mean better quality. Valid range depends on the encoder (e.g.  0-51 for libx264).
                Defaults to None (which will use encoder's default).
                See :ref:`crf` for details.
            preset (str or int, optional): Encoder option that controls the tradeoff between
                encoding encoding speed and compression (output size). Valid on the encoder (commonly
                a string: "fast", "medium", "slow"). Defaults to None
                (which will use encoder's default).
                See :ref:`preset` for details.
            extra_options (dict[str, Any], optional): A dictionary of additional
                encoder options to pass, e.g. ``{"qp": 5, "tune": "film"}``.
                See :ref:`extra_options` for details.

        Returns:
            Tensor: The raw encoded bytes as 1D uint8 Tensor.
        """
        preset_value = str(preset) if isinstance(preset, int) else preset
        return _core.encode_video_to_tensor(
            frames=self._frames,
            frame_rate=self._frame_rate,
            format=format,
            codec=codec,
            pixel_format=pixel_format,
            crf=crf,
            preset=preset_value,
            extra_options=[
                str(x) for k, v in (extra_options or {}).items() for x in (k, v)
            ],
        )

    def to_file_like(
        self,
        file_like,
        format: str,
        *,
        codec: str | None = None,
        pixel_format: str | None = None,
        crf: int | float | None = None,
        preset: str | int | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> None:
        """Encode frames into a file-like object.

        Args:
            file_like: A file-like object that supports ``write()`` and
                ``seek()`` methods, such as io.BytesIO(), an open file in binary
                write mode, etc. Methods must have the following signature:
                ``write(data: bytes) -> int`` and ``seek(offset: int, whence:
                int = 0) -> int``.
            format (str): The container format of the encoded frames, e.g. "mp4", "mov",
                "mkv", "avi", "webm", "flv", etc.
            codec (str, optional): The codec to use for encoding (e.g., "libx264",
                "h264"). If not specified, the default codec
                for the container format will be used.
                See :ref:`codec_selection` for details.
            pixel_format (str, optional): The pixel format for encoding (e.g.,
                "yuv420p", "yuv444p"). If not specified, uses codec's default format.
                See :ref:`pixel_format` for details.
            crf (int or float, optional): Constant Rate Factor for encoding quality. Lower values
                mean better quality. Valid range depends on the encoder (e.g.  0-51 for libx264).
                Defaults to None (which will use encoder's default).
                See :ref:`crf` for details.
            preset (str or int, optional): Encoder option that controls the tradeoff between
                encoding encoding speed and compression (output size). Valid on the encoder (commonly
                a string: "fast", "medium", "slow"). Defaults to None
                (which will use encoder's default).
                See :ref:`preset` for details.
            extra_options (dict[str, Any], optional): A dictionary of additional
                encoder options to pass, e.g. ``{"qp": 5, "tune": "film"}``.
                See :ref:`extra_options` for details.
        """
        preset = str(preset) if isinstance(preset, int) else preset
        _core.encode_video_to_file_like(
            frames=self._frames,
            frame_rate=self._frame_rate,
            format=format,
            file_like=file_like,
            codec=codec,
            pixel_format=pixel_format,
            crf=crf,
            preset=preset,
            extra_options=[
                str(x) for k, v in (extra_options or {}).items() for x in (k, v)
            ],
        )
