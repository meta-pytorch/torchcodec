# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from torchcodec._core.ops import _blocks_frame_metadata, _blocks_frame_planes


class _Metadata(NamedTuple):
    """The fields of the `_blocks_frame_metadata` op, in order."""

    pix_fmt: str
    colorspace: str
    color_range: str
    bit_depth: int
    width: int
    height: int
    rotation_degrees: float


class Packet:
    """Opaque, thread-movable handle to a demuxed (compressed) packet.

    Produced by :class:`Demuxer`, consumed by :class:`VideoPacketDecoder`. It wraps a raw
    pointer, so it is only valid within the process that created it (it cannot
    cross a process boundary).

    Attributes:
        stream_index (int): The index of the stream this packet belongs to,
            absolute across all media types. This is what routes a packet to
            its decoder when it comes out of a :class:`Demuxer` following more
            than one stream.
    """

    def __init__(
        self,
        handle: torch.Tensor,
        stream_index: int,
        *,
        generation: int = 0,
    ):
        self._handle = handle
        self.stream_index = stream_index
        # Which side of the demuxer's last seek this packet came from. Private:
        # it exists so a decoder can catch a missing reset(), not for callers to
        # reason about. See _BasePacketDecoder.decode().
        self._generation = generation


class RawFrame:
    """A decoded (YUV) frame, as the decoder produced it: an opaque,
    thread-movable handle to the frame plus everything describing it.

    Produced by :class:`VideoPacketDecoder`, consumed by :class:`ColorConverter`. The
    handle wraps a raw pointer and is process-local. ``pts_seconds`` and
    ``duration_seconds`` are stamped by the decoder (which knows the stream time
    base) and carried here so the :class:`ColorConverter` need not be bound to
    any stream.

    Every field describes the frame as it was decoded, with no conversion
    applied. In particular ``width``, ``height`` and :attr:`planes` are all
    pre-rotation: :attr:`rotation_degrees` is what ``ColorConverter`` applies
    for you, and what you have to apply yourself if you color-convert the planes
    on your own.

    The samples live on the device of the decoder that produced the frame. A
    CUDA decoder falls back to CPU decoding for streams NVDEC can't handle, but
    it uploads those frames before handing them out, so they are
    indistinguishable from NVDEC ones here.

    All the fields but ``pts_seconds`` and ``duration_seconds`` are computed on
    first access and then cached, so nothing is paid for by a pipeline that only
    color-converts.
    """

    def __init__(
        self,
        handle: torch.Tensor,
        pts_seconds: float,
        duration_seconds: float,
        storage: torch.Tensor | None = None,
    ):
        self._handle = handle
        self._storage = storage
        self.pts_seconds = pts_seconds
        self.duration_seconds = duration_seconds
        self._metadata: _Metadata | None = None
        self._planes: tuple[torch.Tensor, ...] | None = None

    @property
    def _device(self) -> torch.device:
        return (
            self._storage.device if self._storage is not None else torch.device("cpu")
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Tell the CUDA allocator that ``stream`` is still reading this frame's
        samples, and that its memory must not be reused until that work is done.

        **A CUDA consumer reading the frame on a stream other than the one the
        decoder ran on must call this**, right after enqueueing the reads, or
        the decoder's next frame may be handed the same buffer and overwrite it
        while those reads are still queued. :class:`ColorConverter` does it for
        you; anything else you write does not.

        A no-op for frames that aren't on a CUDA device.
        """
        # See [Standalone Frame Storage and the need for record_stream]
        if self._storage is not None:
            self._storage.record_stream(stream)

    def _get_metadata(self) -> _Metadata:
        if self._metadata is None:
            self._metadata = _Metadata(*_blocks_frame_metadata(self._handle))
        return self._metadata

    @property
    def pix_fmt(self) -> str:
        """FFmpeg pixel-format name. On CPU this is the source's own format,
        e.g. ``"yuv420p"``. On CUDA it is always an NVDEC surface format:
        ``"nv12"``, ``"p010le"``, ``"p012le"``, ``"p016le"``, ``"yuv444p"`` or
        ``"yuv444p16le"``.
        """
        return self._get_metadata().pix_fmt

    @property
    def colorspace(self) -> str:
        """e.g. ``"bt709"``."""
        return self._get_metadata().colorspace

    @property
    def color_range(self) -> str:
        """``"tv"`` (limited) or ``"pc"`` (full)."""
        return self._get_metadata().color_range

    @property
    def bit_depth(self) -> int:
        """The depth of :attr:`pix_fmt` (always), which is the source's bit
        depth except where a CUDA surface format's container is wider than the
        source samples: a 10-bit 4:4:4 source is uploaded as ``yuv444p16le``,
        and a 12-bit source is tagged ``p016le`` on FFmpeg < 6 (which lacks
        ``p012le``). Both report 16 here (on CPU they'd report 10 and 12).

        Everything downstream still reads right, because those samples are
        msb-aligned and are therefore genuinely valid 16-bit ones, with zeroed
        low bits.
        """
        return self._get_metadata().bit_depth

    @property
    def width(self) -> int:
        """Width of the decoded samples, before rotation."""
        return self._get_metadata().width

    @property
    def height(self) -> int:
        """Height of the decoded samples, before rotation."""
        return self._get_metadata().height

    @property
    def rotation_degrees(self) -> float:
        """Degrees counter-clockwise the frame needs to be rotated by to be
        upright, or 0 if the container asks for no rotation. It is *not*
        applied to :attr:`planes`; :class:`ColorConverter` applies it (rounded
        to the nearest multiple of 90) to its output.
        """
        return self._get_metadata().rotation_degrees

    @property
    def planes(self) -> tuple[torch.Tensor, ...]:
        """The decoder's own samples as 2D ``[height, width]`` tensor views,
        with no copy and no conversion: exactly one per component of
        :attr:`pix_fmt`, in the order that format describes. So ``yuv420p`` and
        ``nv12`` both give three, ``yuva420p`` four and ``gray`` one, and the
        chroma ones are subsampled to whatever the format says.

        A component is not a plane: the semi-planar formats (``nv12``,
        ``p010le``, and the rest of the NVDEC surface formats) store U and V
        interleaved in a single allocation, and they come back here as two
        **non-contiguous** views into it. Everything that reads a tensor handles
        that, but a consumer that needs contiguous memory pays a copy for it.

        Raises for the pixel formats that can't be viewed without a copy
        (sub-byte-packed, palettised and float ones) - check :attr:`pix_fmt`
        first if you're decoding something exotic.
        """
        if self._planes is None:
            planes = _blocks_frame_planes(self._handle, self._device)
            # The op's schema is fixed-arity, so it pads with empty tensors up
            # to the 4 components of the widest format.
            self._planes = tuple(p for p in planes if p.numel() > 0)
        return self._planes


# TODO_API_BREAKDOWN DESIGN P1: API design - the class name, and whether
# sample_format is worth carrying now that the layout it describes has been
# normalized away.
@dataclass
class RawAudioSamples:
    """One decoded audio frame's samples, as the decoder produced them.

    Produced by :class:`AudioPacketDecoder`, consumed by
    :class:`AudioConverter`. This is the audio counterpart of
    :class:`RawFrame`, and like it, nothing has been converted: the samples are
    in the codec's own sample type.

    It is not a handle, unlike :class:`RawFrame`. Audio frames are a few kB, so
    the samples are copied out of the ``AVFrame`` rather than viewed, which
    also lets ``[num_channels, num_samples]`` be the layout for every format:
    planar ones store each channel in its own allocation and packed ones
    interleave them, so neither is that shape as it stands.

    Attributes:
        data (torch.Tensor): ``[num_channels, num_samples]``, in the dtype that
            holds the source's samples exactly: ``uint8`` for ``u8``, ``int16``
            for ``s16``, ``int32`` for ``s32``, ``float32`` for ``flt``,
            ``float64`` for ``dbl``. Note the integer ones are *not* normalized
            to ``[-1, 1]``; :class:`AudioConverter` is what does that.
        sample_rate (int): The source's sample rate, in Hz.
        sample_format (str): FFmpeg sample-format name, e.g. ``"s16p"`` or
            ``"fltp"``. This is the format the samples were decoded in, kept
            for provenance: the trailing ``p`` (planar) no longer describes
            :attr:`data`, whose layout is always the same.
        pts_seconds (float): Presentation timestamp of the first sample.
        duration_seconds (float): How long these samples last.
    """

    data: torch.Tensor
    sample_rate: int
    sample_format: str
    pts_seconds: float
    duration_seconds: float
    # See Packet._generation.
    _generation: int = 0

    @property
    def num_channels(self) -> int:
        return self.data.shape[0]

    @property
    def num_samples(self) -> int:
        return self.data.shape[1]
