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

# TODO_API_BREAKDOWN CORRECTNESS P1: Need to understand the seeking we do: it's
# not completley approximate and it's not completely exact either. Understand
# it, document it (?), test it. Maybe we acutally do want to replicate
# approximate and exact. Might not actually be difficult.


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

    def seek(self, seconds: float) -> None:
        """Seek to a keyframe near ``seconds``.

        The next packet returned is that keyframe's, so the frames decoded
        right after a seek typically *precede* ``seconds``: it's up to the
        caller to drop the ones it doesn't want, by looking at their
        ``pts_seconds``. Decoding can't start anywhere else, since every frame
        up to the target is needed to reconstruct it.

        Seeking is approximate, with exactly the semantics of
        ``VideoDecoder(..., seek_mode="approximate")``: we hand FFmpeg the
        target and take whatever keyframe it lands on. That is usually the
        keyframe preceding ``seconds``, but on streams whose keyframes are
        reordered it can be one displayed *after* it, in which case the frames
        in between are unreachable. Landing exactly requires an index of our
        own, which ``VideoDecoder``'s ``seek_mode="exact"`` scans the file to
        build.

        This does not touch any :class:`PacketDecoder`: their buffered
        reference frames are stale afterwards, and feeding them post-seek
        packets without calling :meth:`PacketDecoder.reset` first produces
        corrupt frames.
        """
        _blocks_demuxer_seek(self._handle, float(seconds))

    def __iter__(self):
        while True:
            packet = self.next_packet()
            if packet is None:
                return
            yield packet
