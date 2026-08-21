# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
from pathlib import Path

from torch import Tensor

from torchcodec._core._decoder_utils import create_demuxer
from torchcodec._core.ops import _blocks_demuxer_next_packet, _blocks_demuxer_seek

from ._frame import Packet

# TODO_API_BREAKDOWN FEAT PERF Do we want / need to support 'batch-like' APIs
# were containers are pre-allocated for perf? Like if a user wants to decode
# specific timestamps for sampling?


class Demuxer:
    """Demux building block: opens a container and yields the compressed
    :class:`Packet`\\ s for one (video) stream. Does no decoding.

    This block is passive (it does no threading of its own) and is *not*
    thread-safe: use one ``Demuxer`` per thread. It streams from the start of
    the file, or from wherever :meth:`seek` left it.

    A :class:`Demuxer` also carries the stream configuration used to build a
    :class:`PacketDecoder` and :class:`ColorConverter`, so those are constructed from
    a demuxer and no extra container is opened.

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
            media types. It must refer to a video stream: audio streams aren't
            supported, and requesting one raises an error. If left unspecified,
            then the :term:`best stream` is used.
    """

    def __init__(
        self,
        source: str | Path | bytes | Tensor | io.RawIOBase | io.BufferedReader,
        *,
        stream_index: int | None = None,
    ):
        self._handle = create_demuxer(source=source, stream_index=stream_index)

    def next_packet(self) -> Packet | None:
        """Return the next :class:`Packet`, or ``None`` at end of stream."""
        handle, is_eof = _blocks_demuxer_next_packet(self._handle)
        return None if is_eof else Packet(handle)

    # TODO_API_BREAKDOWN FEAT P1: this is VideoDecoder's "approximate"
    # seek mode (with get_frame_played_at() only). Do we want to offer an
    # "exact" one? It needs a presentation-order keyframe index, i.e. a scan of
    # the whole file, which a user would have to opt into (a Demuxer.scan()?).
    def seek(self, seconds: float) -> None:
        _blocks_demuxer_seek(self._handle, float(seconds))

    def __iter__(self):
        while True:
            packet = self.next_packet()
            if packet is None:
                return
            yield packet
