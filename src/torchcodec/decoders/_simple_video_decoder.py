import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple, Union

from torch import Tensor

from torchcodec.decoders import _core as core


@dataclass
class Frame(Iterable):
    """A single video frame with associated metadata."""

    data: Tensor
    """The frame data as (3-D ``torch.Tensor``)."""
    pts_seconds: float
    """The :term:`pts` of the frame, in seconds (float)."""
    duration_seconds: float
    """The duration of the frame, in seconds (float)."""

    def __iter__(self) -> Iterator[Union[Tensor, float]]:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)


@dataclass
class FrameBatch(Iterable):
    """Multiple video frames with associated metadata."""

    data: Tensor
    """The frames data as (4-D ``torch.Tensor``)."""
    pts_seconds: Tensor
    """The :term:`pts` of the frame, in seconds (1-D ``torch.Tensor`` of floats)."""
    duration_seconds: Tensor
    """The duration of the frame, in seconds (1-D ``torch.Tensor`` of floats)."""

    def __iter__(self) -> Iterator[Union[Tensor, float]]:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)


_ERROR_REPORTING_INSTRUCTIONS = """
This should never happen. Please report an issue following the steps in <TODO_UPDATE_LINK>.
"""


class SimpleVideoDecoder:
    """A single-stream video decoder.

    Args:
        source (str, ``Pathlib.path``, ``torch.Tensor``, or bytes): The source of the video.

            - If ``str`` or ``Pathlib.path``: a path to a local video file.
            - If ``bytes`` object or ``torch.Tensor``: the raw encoded video data.

    Attributes:
        metadata (StreamMetadata): Metadata of the video stream.
    """

    def __init__(self, source: Union[str, Path, bytes, Tensor]):
        # TODO_BEFORE_RELEASE: Add parameter for dimension order.
        # TODO_BEFORE_RELEASE: Document default dimension order (regardless of whether we add the parameter).
        if isinstance(source, str):
            self._decoder = core.create_from_file(source)
        elif isinstance(source, Path):
            self._decoder = core.create_from_file(str(source))
        elif isinstance(source, bytes):
            self._decoder = core.create_from_bytes(source)
        elif isinstance(source, Tensor):
            self._decoder = core.create_from_tensor(source)
        else:
            raise TypeError(
                f"Unknown source type: {type(source)}. "
                "Supported types are str, Path, bytes and Tensor."
            )
        core.scan_all_streams_to_update_metadata(self._decoder)
        core.add_video_stream(self._decoder)

        self.metadata, self._stream_index = _get_and_validate_stream_metadata(
            self._decoder
        )

        if self.metadata.num_frames_computed is None:
            raise ValueError(
                "The number of frames is unknown. " + _ERROR_REPORTING_INSTRUCTIONS
            )
        self._num_frames = self.metadata.num_frames_computed

        if self.metadata.min_pts_seconds is None:
            raise ValueError(
                "The minimum pts value in seconds is unknown. "
                + _ERROR_REPORTING_INSTRUCTIONS
            )
        self._min_pts_seconds = self.metadata.min_pts_seconds

        if self.metadata.max_pts_seconds is None:
            raise ValueError(
                "The maximum pts value in seconds is unknown. "
                + _ERROR_REPORTING_INSTRUCTIONS
            )
        self._max_pts_seconds = self.metadata.max_pts_seconds

    def __len__(self) -> int:
        return self._num_frames

    def _getitem_int(self, key: int) -> Tensor:
        assert isinstance(key, int)

        if key < 0:
            key += self._num_frames
        if key >= self._num_frames or key < 0:
            raise IndexError(
                f"Index {key} is out of bounds; length is {self._num_frames}"
            )

        frame_data, *_ = core.get_frame_at_index(
            self._decoder, frame_index=key, stream_index=self._stream_index
        )
        return frame_data

    def _getitem_slice(self, key: slice) -> Tensor:
        assert isinstance(key, slice)

        start, stop, step = key.indices(len(self))
        frame_data, *_ = core.get_frames_in_range(
            self._decoder,
            stream_index=self._stream_index,
            start=start,
            stop=stop,
            step=step,
        )
        return frame_data

    def __getitem__(self, key: Union[int, slice]) -> Tensor:
        """TODO_BEFORE_RELEASE: Nicolas Document this, looks like our template doesn't show it, aaarrgghhh"""
        if isinstance(key, int):
            return self._getitem_int(key)
        elif isinstance(key, slice):
            return self._getitem_slice(key)

        raise TypeError(
            f"Unsupported key type: {type(key)}. Supported types are int and slice."
        )

    def get_frame_at(self, index: int) -> Frame:
        """Return a single frame at the given index.

        Args:
            index (int): The index of the frame to retrieve.

        Returns:
            Frame: The frame at the given index.
        """

        if not 0 <= index < self._num_frames:
            raise IndexError(
                f"Index {index} is out of bounds; must be in the range [0, {self._num_frames})."
            )
        frame = core.get_frame_at_index(
            self._decoder, frame_index=index, stream_index=self._stream_index
        )
        return Frame(*frame)

    def get_frames_at(self, start: int, stop: int, step: int = 1) -> FrameBatch:
        """Return multiple frames at the given index range.

        Frames are in [start, stop).

        Args:
            start (int): Index of the first frame to retrieve.
            stop (int): End of indexing range (exclusive, as per Python
                conventions).
            step (int, optional): Step size between frames. Default: 1.

        Returns:
            FrameBatch: The frames within the specified range.
        """
        if not 0 <= start < self._num_frames:
            raise IndexError(
                f"Start index {start} is out of bounds; must be in the range [0, {self._num_frames})."
            )
        if stop < start:
            raise IndexError(
                f"Stop index ({stop}) must not be less than the start index ({start})."
            )
        if not step > 0:
            raise IndexError(f"Step ({step}) must be greater than 0.")
        frames = core.get_frames_in_range(
            self._decoder,
            stream_index=self._stream_index,
            start=start,
            stop=stop,
            step=step,
        )
        return FrameBatch(*frames)

    def get_frame_displayed_at(self, pts_seconds: float) -> Frame:
        """Return a single frame displayed at the given :term:`pts`, in seconds.

        Args:
            pts (float): The :term:`pts` of the frame to retrieve, in seconds.

        Returns:
            Frame: The frame at the given :term:`pts`.
        """
        if not self._min_pts_seconds <= pts_seconds < self._max_pts_seconds:
            raise IndexError(
                f"Invalid pts in seconds: {pts_seconds}. "
                f"It must be greater than or equal to {self._min_pts_seconds} "
                f"and less than or equal to {self._max_pts_seconds}."
            )
        frame = core.get_frame_at_pts(self._decoder, pts_seconds)
        return Frame(*frame)

    def get_frames_displayed_at(
        self, start_seconds: float, stop_seconds: float
    ) -> FrameBatch:
        if not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. It must be less than or equal to stop seconds ({stop_seconds})."
            )
        if not self._min_pts_seconds <= start_seconds < self._max_pts_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be greater than or equal to {self._min_pts_seconds} "
                f"and less than or equal to {self._max_pts_seconds}."
            )
        if not stop_seconds <= self._max_pts_seconds:
            raise ValueError(
                f"Invalid stop seconds: {stop_seconds}. "
                f"It must be less than or equal to {self._max_pts_seconds}."
            )
        frames = core.get_frames_by_pts_in_range(
            self._decoder,
            stream_index=self._stream_index,
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
        )
        return FrameBatch(*frames)


def _get_and_validate_stream_metadata(
    decoder: Tensor,
) -> Tuple[core.StreamMetadata, int]:
    video_metadata = core.get_video_metadata(decoder)

    best_stream_index = video_metadata.best_video_stream_index
    if best_stream_index is None:
        raise ValueError(
            "The best video stream is unknown. " + _ERROR_REPORTING_INSTRUCTIONS
        )

    best_stream_metadata = video_metadata.streams[best_stream_index]
    return (best_stream_metadata, best_stream_index)
