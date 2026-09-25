# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from torchcodec._core._decoder_utils import create_demuxer
from torchcodec._core._metadata import (
    _stream_metadata_from_dict,
    AudioStreamHeaderMetadata,
    ContainerMetadata,
    DemuxerMetadata,
    StreamMetadata,
    VideoStreamHeaderMetadata,
)
from torchcodec._core.ops import (
    _blocks_demuxer_add_stream,
    _blocks_demuxer_container_json_metadata,
    _blocks_demuxer_get_audio_video_stream_indices,
    _blocks_demuxer_next_packet,
    _blocks_demuxer_scan,
    _blocks_demuxer_seek,
    _blocks_demuxer_stream_json_metadata,
)

from .._decoder_utils import convert_device_to_str

from ._frame import Packet
from ._helpers import _process_local
from ._packet_decoder import AudioPacketDecoder, VideoPacketDecoder

# TODO_API_BREAKDOWN FEAT PERF Do we want / need to support 'batch-like' APIs
# were containers are pre-allocated for perf? Like if a user wants to decode
# specific timestamps for sampling?


@dataclass
class FrameIndex:
    """What the packets of a video stream say about its frames, as returned by
    a :term:`scan` (:meth:`VideoStream.scan`).

    Its members come in three shapes:

    - **Per-frame tensors**, of shape ``[N]`` where ``N`` is the number of
      frames: one entry per frame, in presentation order. These are
      :attr:`is_key_frame`, :attr:`pts_seconds` and :attr:`duration_seconds`.
    - **Keyframe positions**, :attr:`key_frame_indices`, of shape ``[K]`` where
      ``K`` is the number of keyframes. These are indices into the per-frame
      tensors, not frames themselves.
    - **Stream-level metadata scalars**, one value for the whole stream:
      :attr:`num_frames_from_content`,
      :attr:`begin_stream_seconds_from_content`,
      :attr:`end_stream_seconds_from_content` and
      :attr:`average_fps_from_content`.

    :meth:`index_at` and :meth:`key_frame_seconds_for` are convenience methods
    that search those tensors so that you don't have to. Together with
    :meth:`Demuxer.seek`, they are what an exact seek is built from::

        frame_index = video_stream.scan()

        # Reach the frame displayed at 12.5 seconds
        target = frame_index.pts_seconds[frame_index.index_at(12.5)]
        demuxer.seek(frame_index.key_frame_seconds_for(target))
        packet_decoder.reset()
        # ... then decode forward, dropping the frames before `target`

    Everything here is derived from the stream's packets rather than from the
    container header, so it is exact where the header is only a claim. That is
    what the ``_from_content`` suffixes mark, against the ``_from_header`` ones
    on :attr:`VideoStream.metadata`.
    """

    is_key_frame: Tensor
    """Bool tensor of shape ``[N]``, whether each frame is a keyframe."""
    _pts: Tensor
    _duration: Tensor
    _time_base_num: int
    _time_base_den: int

    def __len__(self) -> int:
        """The number of frames in the stream."""
        return self.is_key_frame.shape[0]

    @property
    def num_frames_from_content(self) -> int:
        """The number of frames in the stream."""
        return len(self)

    @cached_property
    def pts_seconds(self) -> Tensor:
        """Float64 tensor of shape ``[N]``, the :term:`pts` of each frame."""
        return self._to_seconds(self._pts)

    @cached_property
    def duration_seconds(self) -> Tensor:
        """Float64 tensor of shape ``[N]``, how long each frame is displayed
        for."""
        return self._to_seconds(self._duration)

    @cached_property
    def key_frame_indices(self) -> Tensor:
        """Int64 tensor of shape ``[K]``, the indices of the keyframes."""
        return self.is_key_frame.nonzero().squeeze(1)

    @cached_property
    def begin_stream_seconds_from_content(self) -> float:
        """The :term:`pts` of the first frame."""
        return float(self.pts_seconds[0])

    @cached_property
    def end_stream_seconds_from_content(self) -> float:
        """The time at which the last frame stops being displayed.

        This is the largest ``pts + duration`` across the stream, not the last
        frame's own end time. Durations vary, so the frame that finishes last
        isn't necessarily the one that starts last.
        """
        return float(self._end_seconds_so_far[-1])

    @property
    def average_fps_from_content(self) -> float:
        """The average number of frames per second over the stream."""
        return len(self) / (
            self.end_stream_seconds_from_content
            - self.begin_stream_seconds_from_content
        )

    def index_at(self, seconds: float | Tensor) -> int | Tensor:
        """The index of the frame being displayed at ``seconds``.

        A frame is displayed from its own :term:`pts` until that plus its
        duration, so this is the frame whose interval contains ``seconds``.

        Args:
            seconds (float or Tensor): The timestamp(s) to look up. A value
                outside the stream gives the closest frame, i.e. the first or
                the last one.

        Returns:
            int or Tensor: The index of that frame. A tensor of timestamps
            gives an int64 tensor of indices of the same shape.
        """
        # First frame that hasn't finished playing by `seconds`, which is
        # get_frame_played_at()'s criterion (frame_start <= t < frame_end,
        # SingleStreamDecoder.cpp) expressed as a search rather than a scan of
        # decoded frames. Note it is *not* seconds_to_index_lower_bound(), which
        # compares against next_pts and so answers differently for a timestamp
        # falling in a gap between two frames.
        indices = torch.searchsorted(
            self._end_seconds_so_far,
            torch.as_tensor(seconds, dtype=torch.float64),
            right=True,
        ).clamp(max=len(self) - 1)
        return indices if isinstance(seconds, Tensor) else int(indices)

    # TODO_API_BREAKDOWN DESIGN P1: Still kinda hate this name
    def key_frame_seconds_for(self, seconds: float) -> float:
        """The timestamp of the last keyframe at or before ``seconds``.

        Args:
            seconds (float): The timestamp you want to reach.

        Returns:
            float: The timestamp to seek to.
        """
        # Same search as get_key_frame_index_for_pts_using_scanned_index()
        # (SingleStreamDecoder.cpp): upper_bound minus one, i.e. the last
        # keyframe at or before the target, and -1 when there is none.
        position = (
            int(torch.searchsorted(self._key_frame_seconds, seconds, right=True)) - 1
        )
        if position < 0:
            # No keyframe at or before the target: the one it decodes from was
            # trimmed away by an edit list, so it isn't in this index. Aim at
            # the start of the stream and let FFmpeg find it, which is what
            # exact mode does when that search returns -1.
            return float(self.pts_seconds[0])
        return float(self._key_frame_seconds[position])

    def _to_seconds(self, value: Tensor) -> Tensor:
        # pts_to_seconds() (FFMPEGCommon.cpp), and it has to stay bit-for-bit
        # identical to it: float64 is C++ `double`, and the multiplication comes
        # before the division on both sides. Timestamps from a FrameIndex are
        # compared against, and fed back into, values the C++ produced.
        return value.to(torch.float64) * self._time_base_num / self._time_base_den

    @cached_property
    def _end_seconds_so_far(self) -> Tensor:
        # A running max, because the raw end times aren't necessarily sorted: a
        # frame whose duration overruns the start of the next one finishes after
        # it. We need this to be sorted for the binary search in index_at() to
        # work.
        return self._to_seconds(self._pts + self._duration).cummax(dim=0).values

    @cached_property
    def _key_frame_seconds(self) -> Tensor:
        return self.pts_seconds[self.key_frame_indices]


@_process_local("Reach it through a Demuxer built in that process.")
class _Stream:
    _media_type: str

    def __init__(self, demuxer: Demuxer, index: int):
        self._demuxer = demuxer
        self.index = index

    def _read_metadata(self) -> StreamMetadata:
        return _stream_metadata_from_dict(
            json.loads(
                _blocks_demuxer_stream_json_metadata(self._demuxer._handle, self.index)
            ),
            self.index,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self.index})"


class VideoStream(_Stream):
    """A video stream followed by a :class:`Demuxer`.

    You should not build one yourself: a ``Demuxer`` creates one for you.

    Attributes:
        index (int): The stream's index within the container, absolute across
            all media types. This is what a :class:`Packet`'s ``stream_index``
            can be compared against.
    """

    _media_type = "video"

    def __init__(self, demuxer: Demuxer, index: int):
        super().__init__(demuxer, index)
        self._frame_index: FrameIndex | None = None

    @cached_property
    def metadata(self) -> VideoStreamHeaderMetadata:
        """What the container header says about this video stream.

        From the header only. This stream's exact frame count, timestamps and
        keyframe positions aren't in there - those come from :meth:`scan`, and
        that is the distinction the ``_from_header`` and ``_from_content``
        suffixes mark.
        """
        return cast(VideoStreamHeaderMetadata, self._read_metadata())

    def scan(self) -> FrameIndex:
        """Demux this stream from end to end, without decoding it (a
        :term:`scan`), and return its :class:`FrameIndex`.

        This is the only way to know a stream's exact frame count, timestamps
        and keyframe positions: the container header can be wrong about all
        three.

        You can call this on multiple video streams from the same demuxer. The
        first call scans the entire video once, and subsequent calls on other
        streams are free.

        .. important::

            If called, then this must be called before any packets are read from
            the demuxer.

        Returns:
            FrameIndex: the frame index for this stream, describing its keyframes and frame positions.
        """
        if self._frame_index is None:
            pts, duration, is_key_frame, time_base_num, time_base_den = (
                _blocks_demuxer_scan(self._demuxer._handle, self.index)
            )
            self._frame_index = FrameIndex(
                is_key_frame=is_key_frame,
                _pts=pts,
                _duration=duration,
                _time_base_num=time_base_num,
                _time_base_den=time_base_den,
            )
        return self._frame_index

    def make_decoder(
        self,
        device: str | torch.device | None = None,
        *,
        num_ffmpeg_threads: int = 1,
    ) -> VideoPacketDecoder:
        """Build the :class:`VideoPacketDecoder` for this stream.

        Args:
            device (str or torch.device, optional): The device to decode on (cpu or CUDA).
                If ``None`` (default), the current default device is used (see
                ``torch.set_default_device``).
            num_ffmpeg_threads (int, optional): The number of threads to use for
                CPU decoding. This has no effect when decoding on GPU. Use 1 for
                single-threaded decoding, which may be best if you are decoding
                multiple streams in parallel. Use a higher number for
                multi-threaded decoding, which is best for a single stream.
                Passing 0 lets FFmpeg decide on the number of threads.
                Default: 1.

        Returns:
            VideoPacketDecoder: A decoder for this stream's packets.
        """
        if num_ffmpeg_threads is None:
            raise ValueError(f"{num_ffmpeg_threads = } should be an int.")
        return VideoPacketDecoder._from_stream(
            self, convert_device_to_str(device), num_ffmpeg_threads
        )


class AudioStream(_Stream):
    """An audio stream followed by a :class:`Demuxer`.

    You should not build one yourself: a ``Demuxer`` creates one for you.

    Attributes:
        index (int): The stream's index within the container, absolute across
            all media types. This is what a :class:`Packet`'s ``stream_index``
            can be compared against.
    """

    _media_type = "audio"

    @cached_property
    def metadata(self) -> AudioStreamHeaderMetadata:
        """What the container header says about this audio stream.

        From the header only.
        """
        return cast(AudioStreamHeaderMetadata, self._read_metadata())

    def make_decoder(self) -> AudioPacketDecoder:
        """Build the :class:`AudioPacketDecoder` for this stream.

        Returns:
            AudioPacketDecoder: A decoder for this stream's packets.
        """
        return AudioPacketDecoder._from_stream(self, "cpu")


@_process_local(
    "It owns an open container and the file descriptor behind it. Construct "
    "one in each process, from the same source."
)
class Demuxer:
    """Reads one or more video and audio streams from a container, and produces their compressed :class:`Packet`\\ s.

    Low-level API: for straightforward decoding, use
    :class:`~torchcodec.decoders.VideoDecoder` or
    :class:`~torchcodec.decoders.AudioDecoder` instead.

    Packets come out interleaved, and :attr:`Packet.stream_index` says which
    stream each one belongs to::

        demuxer = Demuxer("video.mp4", streams=("video", "audio"))
        decoders = {s.index: s.make_decoder() for s in demuxer.streams}

        for packet in demuxer:
            for output in decoders[packet.stream_index].decode(packet):
                ...

    Args:
        source (str, ``Pathlib.path``, bytes, ``torch.Tensor`` or file-like object): The source of the media:

            - If ``str``: a local path or a URL to a media file.
            - If ``Pathlib.path``: a path to a local media file.
            - If ``bytes`` object or ``torch.Tensor``: the raw encoded data.
            - If file-like object: we read data from the object on demand. The
              object must expose the methods `read(self, size: int) -> bytes`
              and `seek(self, offset: int, whence: int) -> int`.
        streams (str, int or tuple, optional): Which streams to follow, as a
            single selector or a tuple of them. A selector is either
            ``"video"`` or ``"audio"`` for the :term:`best stream` of that
            type, or an ``int`` for a stream index, absolute across all media
            types. ``"all"`` follows every audio and video stream in container
            order, skipping the rest, and can only be used on its own. Default:
            ``"video"``.

    Attributes:
        streams (tuple): The :class:`VideoStream` and :class:`AudioStream`
            objects being followed, in the order the ``streams`` parameter
            named them. Packet decoders are built from these.
        metadata (DemuxerMetadata): What the container header says about the
            container itself. What it says about a given stream is on
            ``demuxer.streams[i].metadata``.
    """

    def __init__(
        self,
        source: str | Path | bytes | Tensor | io.RawIOBase | io.BufferedReader,
        *,
        streams: str | int | tuple[str | int, ...] = "video",
    ):
        self._handle = create_demuxer(source=source)
        # _generation is bumped on every seek and stamped on every packet, so
        # that a decoder can tell if the user forgot to `reset()` it.
        self._generation = 0
        try:
            self.streams = tuple(
                self._add_stream(selector) for selector in self._parse_streams(streams)
            )
        except RuntimeError as e:
            # Just to be nice, we convert the C++ RuntimeErrors into ValueError
            # so all stream validation errors are consistently ValueError.
            raise ValueError(str(e)) from None

    @cached_property
    def metadata(self) -> DemuxerMetadata:
        return DemuxerMetadata(**_container_fields(self._handle))

    def _parse_streams(self, streams) -> list[int | str]:
        if streams == "all":
            return [
                int(index)
                for index in _blocks_demuxer_get_audio_video_stream_indices(
                    self._handle
                )
            ]

        if isinstance(streams, (str, int)) or not isinstance(streams, Iterable):
            streams = (streams,)
        streams = tuple(streams)
        if not streams:
            raise ValueError(
                "streams is empty, so this demuxer would have nothing to demux."
            )

        for selector in streams:
            if selector in ("video", "audio"):
                continue
            if selector == "all":
                raise ValueError(
                    "streams='all' can only be used on its own, not alongside "
                    f"other selectors: got {streams!r}."
                )
            if not isinstance(selector, int) or isinstance(selector, bool):
                raise ValueError(
                    f"Invalid stream selector {selector!r}. Expected 'video', "
                    "'audio', or an int stream index."
                )
        return list(streams)

    def _add_stream(self, selector: int | str) -> _Stream:
        index, media_type = _blocks_demuxer_add_stream(
            self._handle,
            selector if isinstance(selector, int) else None,
            selector if isinstance(selector, str) else None,
        )
        stream_class = VideoStream if media_type == "video" else AudioStream
        return stream_class(self, index)

    def __iter__(self) -> Demuxer:
        return self

    def __next__(self) -> Packet:
        """Read and return the next :class:`Packet`.

        Packets come out interleaved across the streams being followed, in the
        order the container stores them, so this is where
        :attr:`Packet.stream_index` matters: it is what routes each packet to
        the decoder of its own stream.

        Returns:
            Packet: The next packet.
        """
        handle, is_eof, stream_index = _blocks_demuxer_next_packet(self._handle)
        if is_eof:
            raise StopIteration
        return Packet(handle, stream_index, generation=self._generation)

    # Note: could consider adding int-based APIs? Not sure if needed for seek
    # but we could at least expose the int-based pts values along with the time
    # base, etc.
    def seek(
        self, seconds: float, *, stream: VideoStream | AudioStream | None = None
    ) -> None:
        """Move the demuxer to ``seconds``.

        This moves *every* stream being followed. For videos, this lands on the
        keyframe at or before ``seconds``. For audio, a lossy codec's first
        frames after a seek are typically slightly wrong until the codec
        re-primes. This is especially true when resampling is involved (via an
        :class:`AudioConverter`). Pre-rolling a margin of audio before the
        target is up to you.

        There is no ``seek_mode`` to choose from: seeking straight to
        ``seconds`` is what :class:`~torchcodec.decoders.VideoDecoder` calls
        ``seek_mode="approximate"``. To get the ``seek_mode="exact"`` behavior,
        :term:`scan` the stream and seek to
        :meth:`FrameIndex.key_frame_seconds_for` of your target instead.

        .. important::

            You must call :meth:`VideoPacketDecoder.reset` or
            :meth:`AudioPacketDecoder.reset` on every decoder fed by this
            demuxer afterwards, and :meth:`AudioConverter.reset` on every
            converter too: a seek invalidates a codec and resampler states.

        Args:
            seconds (float): The position to seek to.
            stream (VideoStream or AudioStream, optional): The stream the
                target ``seconds`` is resolved against. FFmpeg resolves a seek in a single
                stream's time base and lands on *that* stream's keyframes, the
                other streams merely resuming from wherever the container ends
                up - so a second video stream may land mid-GOP and decode
                garbage until its next keyframe. Defaults to the first of
                :attr:`streams`, as passed to the constructor.
        """
        _blocks_demuxer_seek(
            self._handle,
            float(seconds),
            None if stream is None else stream.index,
        )
        self._generation += 1


def _container_fields(handle: Tensor) -> dict:
    container_dict = json.loads(_blocks_demuxer_container_json_metadata(handle))
    return dict(
        duration_seconds_from_header=container_dict.get("durationSecondsFromHeader"),
        bit_rate_from_header=container_dict.get("bitRate"),
        best_video_stream_index=container_dict.get("bestVideoStreamIndex"),
        best_audio_stream_index=container_dict.get("bestAudioStreamIndex"),
    )


def get_container_metadata(
    source: str | Path | bytes | Tensor | io.RawIOBase | io.BufferedReader,
) -> ContainerMetadata:
    """Describe a container and every stream in it, without decoding anything.

    .. code-block:: python

        metadata = get_container_metadata("video.mp4")
        metadata.duration_seconds_from_header
        metadata.streams[metadata.best_video_stream_index].width

    Only the header is read, so this is what to reach for before you know what a
    file holds, and therefore which streams to ask a :class:`Demuxer` for.

    Args:
        source (str, ``Pathlib.path``, bytes, ``torch.Tensor`` or file-like object):
            The source of the media, as for :class:`Demuxer`.

    Returns:
        ContainerMetadata: The container's own metadata, plus one entry per
        stream, indexed by stream index.
    """

    handle = create_demuxer(source=source)
    container_dict = json.loads(_blocks_demuxer_container_json_metadata(handle))
    streams = [
        _stream_metadata_from_dict(
            json.loads(_blocks_demuxer_stream_json_metadata(handle, stream_index)),
            stream_index,
        )
        for stream_index in range(int(container_dict["numStreams"]))
    ]
    return ContainerMetadata(**_container_fields(handle), streams=streams)
