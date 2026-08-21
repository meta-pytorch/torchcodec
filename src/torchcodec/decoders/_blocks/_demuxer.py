# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch
from torch import Tensor

from torchcodec._core._decoder_utils import create_demuxer
from torchcodec._core.ops import (
    _blocks_demuxer_next_packet,
    _blocks_demuxer_scan,
    _blocks_demuxer_seek,
)

from ._frame import Packet

# TODO_API_BREAKDOWN FEAT PERF Do we want / need to support 'batch-like' APIs
# were containers are pre-allocated for perf? Like if a user wants to decode
# specific timestamps for sampling?


# TODO_API_BREAKDOWN DESIGN P1: need to figure out the right API for this.
# StreamIndex? the fields? The methods?
# We'll have a big problem if/when we implement multi-stream demuxing - and
# audio!
@dataclass
class StreamIndex:
    """The content of a video stream, as returned by :meth:`VideoDemuxer.scan`.

    One entry per frame, in presentation order. Everything here is derived from
    the packets of the stream rather than from the container header, so the
    values are exact.

    Timestamps are stored in the stream's own time base and converted to
    seconds on access, which is what makes those conversions exact: a frame's
    end time is ``(pts + duration)`` converted once, not ``pts_seconds +
    duration_seconds``, which rounds twice and can land on the wrong side of a
    frame boundary.

    Attributes:
        is_key_frame (torch.Tensor): bool ``[N]``, whether each frame is a
            :term:`keyframe`.
    """

    is_key_frame: Tensor
    _pts: Tensor
    _duration: Tensor
    _time_base_num: int
    _time_base_den: int

    def __len__(self) -> int:
        """The number of frames in the stream."""
        return self.is_key_frame.shape[0]

    @cached_property
    def pts_seconds(self) -> Tensor:
        """float64 ``[N]``, the presentation timestamp of each frame."""
        return self._to_seconds(self._pts)

    @cached_property
    def duration_seconds(self) -> Tensor:
        """float64 ``[N]``, how long each frame is displayed for."""
        return self._to_seconds(self._duration)

    @cached_property
    def key_frame_indices(self) -> Tensor:
        """int64 ``[K]``, the indices of the :term:`keyframe`\\ s."""
        return self.is_key_frame.nonzero().squeeze(1)

    @cached_property
    def begin_stream_seconds(self) -> float:
        """Presentation timestamp of the first frame."""
        return float(self.pts_seconds[0])

    @cached_property
    def end_stream_seconds(self) -> float:
        """Timestamp at which the last frame stops being displayed."""
        # max(), not [-1]: durations vary, so the frame that finishes last
        # isn't necessarily the one that starts last. This is how
        # end_stream_pts_from_content is accumulated in SingleStreamDecoder.
        return float(self._end_seconds.max())

    @property
    def average_fps(self) -> float:
        """Average number of frames per second over the stream."""
        return len(self) / (self.end_stream_seconds - self.begin_stream_seconds)

    def index_at(self, seconds: float) -> int:
        """Index of the frame that is being displayed at ``seconds``.

        A timestamp outside of the stream gives the frame closest to it, i.e.
        the first or the last one.
        """
        # First frame that hasn't finished playing by `seconds`, which is
        # get_frame_played_at()'s criterion (frame_start <= t < frame_end,
        # SingleStreamDecoder.cpp) expressed as a search rather than a scan of
        # decoded frames. Note it is *not* seconds_to_index_lower_bound(), which
        # compares against next_pts and so answers differently for a timestamp
        # falling in a gap between two frames.
        index = int(torch.searchsorted(self._end_seconds, seconds, right=True))
        return min(index, len(self) - 1)

    def key_frame_seconds_for(self, seconds: float) -> float:
        """Timestamp to :meth:`VideoDemuxer.seek` to in order to reach ``seconds``:
        that of the last :term:`keyframe` which isn't after it.

        This is what makes a seek *exact*. FFmpeg resolves a seek against decode
        timestamps, so on a file whose keyframes are reordered, seeking to the
        target itself can land on a keyframe that is displayed after it, leaving
        the frames in between unreachable
        (https://trac.ffmpeg.org/ticket/11137). Seeking to a keyframe's own
        timestamp lands on that keyframe, and the target is then reached by
        decoding forward from there.

        To reach a frame *index* rather than a timestamp, pass that frame's
        timestamp: ``key_frame_seconds_for(pts_seconds[i])``.
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
        # before the division on both sides. Timestamps from a StreamIndex are
        # compared against, and fed back into, values the C++ produced.
        return value.to(torch.float64) * self._time_base_num / self._time_base_den

    @cached_property
    def _end_seconds(self) -> Tensor:
        return self._to_seconds(self._pts + self._duration)

    @cached_property
    def _key_frame_seconds(self) -> Tensor:
        return self.pts_seconds[self.key_frame_indices]


class _BaseDemuxer:
    """Shared machinery for :class:`VideoDemuxer` and :class:`AudioDemuxer`.

    Subclasses set ``_media_type``, which is what the stream selection is done
    against.
    """

    _media_type: str

    def __init__(
        self,
        source: str | Path | bytes | Tensor | io.RawIOBase | io.BufferedReader,
        *,
        stream_index: int | None = None,
    ):
        self._handle = create_demuxer(
            source=source, stream_index=stream_index, media_type=self._media_type
        )

    def next_packet(self) -> Packet | None:
        """Return the next :class:`Packet`, or ``None`` at end of stream."""
        handle, is_eof = _blocks_demuxer_next_packet(self._handle)
        return None if is_eof else Packet(handle)

    def seek(self, seconds: float) -> None:
        """Move the demuxer to ``seconds``.

        A seek invalidates whatever the decoder is holding on to, so the
        :class:`PacketDecoder` must be ``reset()`` afterwards.

        Where you land, and what comes out first, depends on the medium. For
        video, a decoder can only start on a :term:`keyframe`, so this lands on
        the keyframe at or before the target and the first frames that come out
        usually precede it. Audio has no keyframe to land on: the codec's
        overlap-add state is lost, so a lossy stream's first few frames after a
        seek are subtly wrong - plausible, but not the samples that whole-file
        decoding would give - until it re-primes. Decoding a margin before the
        target and discarding it is the caller's responsibility.
        """
        _blocks_demuxer_seek(self._handle, float(seconds))

    def __iter__(self):
        while True:
            packet = self.next_packet()
            if packet is None:
                return
            yield packet


class VideoDemuxer(_BaseDemuxer):
    """Demux building block: opens a container and yields the compressed
    :class:`Packet`\\ s for one video stream. Does no decoding.

    This block is passive (it does no threading of its own) and is *not*
    thread-safe: use one ``VideoDemuxer`` per thread. It streams from the start
    of the file, or from wherever :meth:`seek` left it.

    A :class:`VideoDemuxer` also carries the stream configuration used to build a
    :class:`PacketDecoder`, so that is constructed from a demuxer and no extra
    container is opened.

    Args:
        source (str, ``Pathlib.path``, bytes, ``torch.Tensor`` or file-like object): The source of the video:

            - If ``str``: a local path or a URL to a video file.
            - If ``Pathlib.path``: a path to a local video file.
            - If ``bytes`` object or ``torch.Tensor``: the raw encoded video data.
            - If file-like object: we read video data from the object on demand. The object must
              expose the methods `read(self, size: int) -> bytes` and
              `seek(self, offset: int, whence: int) -> int`. Note that every
              read has to re-acquire the GIL, so a file-like source doesn't
              parallelize with the rest of a pipeline as well as the others do.

        stream_index (int, optional): Specifies which stream in the video to
            demux packets from. Note that this index is absolute across all
            media types. It must refer to a video stream; use
            :class:`AudioDemuxer` for audio. If left unspecified, then the
            :term:`best stream` is used.
    """

    _media_type = "video"

    def scan(self) -> StreamIndex:
        """Demux the entire stream, without decoding it, and return its
        :class:`StreamIndex`.

        This reads the whole file, and it is the only way to know the stream's
        exact frame count, timestamps and :term:`keyframe` positions: the
        container header can be wrong about all of them.

        The index always covers the entire stream, wherever the demuxer
        currently is, and the demuxer is left back at the start - so a
        :class:`PacketDecoder` built from it must be ``reset()``, as after a
        ``seek()``. Nothing is cached: calling this twice scans twice.
        """
        pts, duration, is_key_frame, time_base_num, time_base_den = (
            _blocks_demuxer_scan(self._handle)
        )
        return StreamIndex(
            is_key_frame=is_key_frame,
            _pts=pts,
            _duration=duration,
            _time_base_num=time_base_num,
            _time_base_den=time_base_den,
        )


class AudioDemuxer(_BaseDemuxer):
    """Demux building block: opens a container and yields the compressed
    :class:`Packet`\\ s for one audio stream. Does no decoding.

    This block is passive (it does no threading of its own) and is *not*
    thread-safe: use one ``AudioDemuxer`` per thread. It streams from the start
    of the file, or from wherever :meth:`seek` left it.

    An :class:`AudioDemuxer` also carries the stream configuration used to build
    a :class:`PacketDecoder`, so that is constructed from a demuxer and no extra
    container is opened.

    Unlike :class:`VideoDemuxer` there is no ``scan()``: a :class:`StreamIndex`
    describes keyframes and frame indices, and audio has neither.

    Args:
        source (str, ``Pathlib.path``, bytes, ``torch.Tensor`` or file-like object): The source of the audio:

            - If ``str``: a local path or a URL to an audio or video file.
            - If ``Pathlib.path``: a path to a local audio or video file.
            - If ``bytes`` object or ``torch.Tensor``: the raw encoded data.
            - If file-like object: we read data from the object on demand. The object must
              expose the methods `read(self, size: int) -> bytes` and
              `seek(self, offset: int, whence: int) -> int`. Note that every
              read has to re-acquire the GIL, so a file-like source doesn't
              parallelize with the rest of a pipeline as well as the others do.

        stream_index (int, optional): Specifies which stream in the file to
            demux packets from. Note that this index is absolute across all
            media types. It must refer to an audio stream; use
            :class:`VideoDemuxer` for video. If left unspecified, then the
            :term:`best stream` is used.
    """

    _media_type = "audio"
