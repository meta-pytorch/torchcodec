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

import torch
from torch import Tensor

from torchcodec._core._decoder_utils import create_demuxer
from torchcodec._core._metadata import (
    _stream_metadata_from_dict,
    ContainerMetadata,
    DemuxerMetadata,
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
from ._packet_decoder import AudioPacketDecoder, VideoPacketDecoder

# TODO_API_BREAKDOWN FEAT PERF Do we want / need to support 'batch-like' APIs
# were containers are pre-allocated for perf? Like if a user wants to decode
# specific timestamps for sampling?


@dataclass
class FrameIndex:
    """TODO_API_BREAKDOWN DOC"""

    is_key_frame: Tensor
    _pts: Tensor
    _duration: Tensor
    _time_base_num: int
    _time_base_den: int

    def __len__(self) -> int:
        return self.is_key_frame.shape[0]

    @property
    def num_frames_from_content(self) -> int:
        return len(self)

    @cached_property
    def pts_seconds(self) -> Tensor:
        return self._to_seconds(self._pts)

    @cached_property
    def duration_seconds(self) -> Tensor:
        return self._to_seconds(self._duration)

    @cached_property
    def key_frame_indices(self) -> Tensor:
        return self.is_key_frame.nonzero().squeeze(1)

    @cached_property
    def begin_stream_seconds_from_content(self) -> float:
        return float(self.pts_seconds[0])

    @cached_property
    def end_stream_seconds_from_content(self) -> float:
        # max(), not [-1]: durations vary, so the frame that finishes last
        # isn't necessarily the one that starts last. This is how
        # end_stream_pts_from_content is accumulated in SingleStreamDecoder.
        return float(self._end_seconds.max())

    @property
    def average_fps_from_content(self) -> float:
        return len(self) / (
            self.end_stream_seconds_from_content
            - self.begin_stream_seconds_from_content
        )

    def index_at(self, seconds: float) -> int:
        # First frame that hasn't finished playing by `seconds`, which is
        # get_frame_played_at()'s criterion (frame_start <= t < frame_end,
        # SingleStreamDecoder.cpp) expressed as a search rather than a scan of
        # decoded frames. Note it is *not* seconds_to_index_lower_bound(), which
        # compares against next_pts and so answers differently for a timestamp
        # falling in a gap between two frames.
        index = int(torch.searchsorted(self._end_seconds, seconds, right=True))
        return min(index, len(self) - 1)

    # TODO_API_BREAKDOWN DESIGN P1: Still kinda hate this name
    def key_frame_seconds_for(self, seconds: float) -> float:
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
    def _end_seconds(self) -> Tensor:
        return self._to_seconds(self._pts + self._duration)

    @cached_property
    def _key_frame_seconds(self) -> Tensor:
        return self.pts_seconds[self.key_frame_indices]


class _Stream:
    _media_type: str

    def __init__(self, demuxer: Demuxer, index: int):
        self._demuxer = demuxer
        self.index = index

    @cached_property
    def metadata(self):
        return _stream_metadata_from_dict(
            json.loads(
                _blocks_demuxer_stream_json_metadata(self._demuxer._handle, self.index)
            ),
            self.index,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self.index})"


class VideoStream(_Stream):
    """TODO_API_BREAKDOWN DOC"""

    _media_type = "video"

    def __init__(self, demuxer: Demuxer, index: int):
        super().__init__(demuxer, index)
        self._frame_index: FrameIndex | None = None

    def scan(self) -> FrameIndex:
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
        self, device: str | torch.device | None = None
    ) -> VideoPacketDecoder:
        return VideoPacketDecoder._from_stream(self, convert_device_to_str(device))


class AudioStream(_Stream):
    """TODO_API_BREAKDOWN DOC"""

    _media_type = "audio"

    def make_decoder(self) -> AudioPacketDecoder:
        return AudioPacketDecoder._from_stream(self, "cpu")


class Demuxer:
    """Reads one or more video and audio streams from a container, and produces their compressed :class:`Packet`\\ s.

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
    """

    def __init__(
        self,
        source: str | Path | bytes | Tensor | io.RawIOBase | io.BufferedReader,
        *,
        streams: str | int | tuple[str | int, ...] = "video",
    ):
        self._handle = create_demuxer(source=source)
        # Bumped on every seek and stamped on every packet, so that a decoder
        # can tell it is being fed packets from a position it was never reset
        # for. See _BasePacketDecoder.decode().
        self._generation = 0
        self.streams = tuple(
            self._add_stream(selector) for selector in self._parse_streams(streams)
        )

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

    def next_packet(self) -> Packet | None:
        handle, is_eof, stream_index = _blocks_demuxer_next_packet(self._handle)
        if is_eof:
            return None
        return Packet(handle, stream_index, generation=self._generation)

    def seek(self, seconds: float, *, stream: _Stream | None = None) -> None:
        _blocks_demuxer_seek(
            self._handle,
            float(seconds),
            None if stream is None else stream.index,
        )
        self._generation += 1

    def __iter__(self):
        while True:
            packet = self.next_packet()
            if packet is None:
                return
            yield packet


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
    """TODO_API_BREAKDOWN DOC"""

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
