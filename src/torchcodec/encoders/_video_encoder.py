from pathlib import Path
from typing import Union

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
        frame_rate (int): The frame rate of the **input** ``frames``. Also defines the encoded **output** frame rate.
    """

    def __init__(self, frames: Tensor, *, frame_rate: int):
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
        dest: Union[str, Path],
    ) -> None:
        """Encode frames into a file.

        Args:
            dest (str or ``pathlib.Path``): The path to the output file, e.g.
                ``video.mp4``. The extension of the file determines the video
                container format.
        """
        _core.encode_video_to_file(
            frames=self._frames,
            frame_rate=self._frame_rate,
            filename=str(dest),
        )

    def to_tensor(
        self,
        format: str,
    ) -> Tensor:
        """Encode frames into raw bytes, as a 1D uint8 Tensor.

        Args:
            format (str): The container format of the encoded frames, e.g. "mp4", "mov",
            "mkv", "avi", "webm", "flv", or "gif"

        Returns:
            Tensor: The raw encoded bytes as 4D uint8 Tensor.
        """
        return _core.encode_video_to_tensor(
            frames=self._frames,
            frame_rate=self._frame_rate,
            format=format,
        )

    def to_file_like(
        self,
        file_like,
        format: str,
    ) -> None:
        """Encode frames into a file-like object.

        Args:
            file_like: A file-like object that supports ``write()`` and
                ``seek()`` methods, such as io.BytesIO(), an open file in binary
                write mode, etc. Methods must have the following signature:
                ``write(data: bytes) -> int`` and ``seek(offset: int, whence:
                int = 0) -> int``.
            format (str): The container format of the encoded frames, e.g. "mp4", "mov",
                "mkv", "avi", "webm", "flv", or "gif".
        """
        _core.encode_video_to_file_like(
            frames=self._frames,
            frame_rate=self._frame_rate,
            format=format,
            file_like=file_like,
        )
