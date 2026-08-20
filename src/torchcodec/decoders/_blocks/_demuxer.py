# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from torchcodec._core.ops import (
    _blocks_create_demuxer,
    _blocks_demuxer_next_packet,
    _blocks_demuxer_seek,
)

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
    """

    # TODO_API_BREAKDOWN FEAT P1: support file-like, bytes etc.
    def __init__(self, source: str | Path, *, stream_index: int | None = None):
        if isinstance(source, Path):
            source = str(source)
        if not isinstance(source, str):
            raise TypeError(
                f"source must be a path (str or pathlib.Path), got {type(source)}"
            )
        self._handle = _blocks_create_demuxer(source, stream_index)

    def next_packet(self) -> Packet | None:
        """Return the next :class:`Packet`, or ``None`` at end of stream."""
        handle, is_eof = _blocks_demuxer_next_packet(self._handle)
        return None if is_eof else Packet(handle)

    # TODO_API_BREAKDOWN CORRECTNESS P1: this is VideoDecoder's "approximate"
    # seek mode. Do we want to offer an "exact" one? It needs a
    # presentation-order keyframe index, i.e. a scan of the whole file, which
    # a user would have to opt into (a Demuxer.scan()?). seek() would then snap
    # the target back to the preceding keyframe in that index, which is all
    # that VideoDecoder's exact mode does.
    def seek(self, seconds: float) -> None:
        _blocks_demuxer_seek(self._handle, float(seconds))

    def __iter__(self):
        while True:
            packet = self.next_packet()
            if packet is None:
                return
            yield packet
