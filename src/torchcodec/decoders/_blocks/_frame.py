# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from torchcodec._core.ops import (
    _blocks_create_packet,
    _blocks_frame_metadata,
    _blocks_frame_planes,
    _blocks_packet_get_fields,
)

from ._helpers import _bytes_to_tensor, _process_local, _tensor_to_bytes


class _Metadata(NamedTuple):
    # The fields of the `_blocks_frame_metadata` op, in order.
    pixel_format: str
    color_space: str
    color_range: str
    color_primaries: str
    color_transfer_characteristic: str
    bit_depth: int
    width: int
    height: int
    rotation: float


# TODO_DEMUXER_FREE: expose the stream's time base in stream.metadata
# (VideoStreamHeaderMetadata / AudioStreamHeaderMetadata). For now, it's only in
# the codec parameters, which the docs tell users not to read as metadata. Then
# replace "(the codec parameters' ``time_base``)" in the docstring below with a
# reference to it.
class Packet:
    """One compressed packet of one stream.

    A :class:`Demuxer` creates them, and a :class:`VideoPacketDecoder` or an
    :class:`AudioPacketDecoder` consumes them. You can also build one from
    compressed data that doesn't come from a ``Demuxer``, with
    :meth:`from_bytes` or :meth:`from_tensor`.

    Packets can be pickled, e.g. to be decoded in another process. The pickles
    are only guaranteed to be readable by the same version of TorchCodec.
    """

    stream_index: int
    """The index of the stream this packet belongs to, absolute across all
    media types. When a demuxer follows more than one stream, this is what
    routes each packet to the right decoder."""

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

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        pts: int | None = None,
        dts: int | None = None,
        duration: int = 0,
        flags: int = 0,
        side_data: dict[str, bytes] | None = None,
        stream_index: int = 0,
    ) -> Packet:
        """Build a packet from compressed data, e.g. read from a database or
        received from the network.

        Only ``data`` is required: most codecs need nothing else to decode it.

        Args:
            data (bytes, bytearray or memoryview): The compressed data of one
                packet. It is copied.
            pts (int, optional): The :term:`pts`, in units of the stream's time
                base. The decoded frames' :attr:`RawFrame.pts` come from it.
                ``None`` (default) if unknown.
            dts (int, optional): The decoding timestamp, in units of the
                stream's time base. ``None`` (default) if unknown.
            duration (int, optional): How long the packet lasts, in units of
                the stream's time base. Default: 0, for unknown.
            flags (int, optional): FFmpeg's ``AV_PKT_FLAG_*`` bits, e.g. 1
                (``AV_PKT_FLAG_KEY``) for a keyframe. ``AV_PKT_FLAG_TRUSTED``
                (8) isn't allowed. Default: 0.
            side_data (dict, optional): Extra data about the packet, keyed by
                FFmpeg's names for its types, e.g. ``"Skip Samples"``.
            stream_index (int, optional): See :attr:`stream_index`. Default: 0.

        Returns:
            Packet: The packet.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"data must be bytes, bytearray or memoryview, got {type(data)}."
            )
        return cls.from_tensor(
            _bytes_to_tensor(data),
            pts=pts,
            dts=dts,
            duration=duration,
            flags=flags,
            side_data=side_data,
            stream_index=stream_index,
        )

    @classmethod
    def from_tensor(
        cls,
        data: torch.Tensor,
        *,
        pts: int | None = None,
        dts: int | None = None,
        duration: int = 0,
        flags: int = 0,
        side_data: dict[str, bytes] | None = None,
        stream_index: int = 0,
    ) -> Packet:
        """Build a packet from compressed data in a tensor.

        Same as :meth:`from_bytes`, except for ``data``.

        Args:
            data (torch.Tensor): The compressed data of one packet, as a
                contiguous 1D uint8 CPU tensor. It is copied.
            pts (int, optional): See :meth:`from_bytes`.
            dts (int, optional): See :meth:`from_bytes`.
            duration (int, optional): See :meth:`from_bytes`.
            flags (int, optional): See :meth:`from_bytes`.
            side_data (dict, optional): See :meth:`from_bytes`.
            stream_index (int, optional): See :meth:`from_bytes`.

        Returns:
            Packet: The packet.
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"data must be a tensor, got {type(data)}.")
        side_data = side_data or {}
        handle = _blocks_create_packet(
            data,
            pts=pts,
            dts=dts,
            duration=duration,
            flags=flags,
            side_data_names=list(side_data),
            side_data_values=[_bytes_to_tensor(v) for v in side_data.values()],
        )
        return cls(handle, stream_index)

    def _get_fields(self) -> dict:
        outputs = _blocks_packet_get_fields(self._handle)
        names = [r.name for r in _blocks_packet_get_fields._schema.returns]
        return dict(zip(names, outputs))

    def __reduce__(self):
        kwargs = {
            "pts": self.pts,
            "dts": self.dts,
            "duration": self.duration,
            "flags": self.flags,
            "side_data": self.side_data,
            "stream_index": self.stream_index,
        }
        # The generation is kept so that a decoder in another process still
        # catches a missing reset() after a seek.
        return (_unpickle_packet, (self.data, kwargs, self._generation))

    @property
    def data(self) -> torch.Tensor:
        """The compressed data, as a 1D uint8 CPU tensor.

        It is a view on the packet's data: writing to it modifies the packet,
        so ``.clone()`` it first.
        """
        return self._get_fields()["data"]

    @property
    def pts(self) -> int | None:
        """The :term:`pts`, in units of the stream's time base. ``None`` if
        unknown."""
        return self._get_fields()["pts"]

    @property
    def dts(self) -> int | None:
        """The decoding timestamp, in units of the stream's time base. ``None``
        if unknown."""
        return self._get_fields()["dts"]

    @property
    def duration(self) -> int:
        """How long the packet lasts, in units of the stream's time base. 0 if
        unknown."""
        return self._get_fields()["duration"]

    @property
    def flags(self) -> int:
        """FFmpeg's ``AV_PKT_FLAG_*`` bits, e.g. ``flags & 1`` is set for
        keyframes."""
        return self._get_fields()["flags"]

    @property
    def side_data(self) -> dict[str, bytes]:
        """Extra data about the packet, keyed by FFmpeg's names for its types,
        e.g. ``"Skip Samples"``."""
        fields = self._get_fields()
        return {
            name: _tensor_to_bytes(value)
            for name, value in zip(
                fields["side_data_names"], fields["side_data_values"]
            )
        }


def _unpickle_packet(data: torch.Tensor, kwargs: dict, generation: int) -> Packet:
    packet = Packet.from_tensor(data, **kwargs)
    packet._generation = generation
    return packet


@_process_local(
    "Its planes are views on FFmpeg's own memory. To send the samples, copy "
    "them out first - tuple(p.clone() for p in raw_frame.planes) - or convert "
    "them with a ColorConverter and send the resulting Frame."
)
class RawFrame:
    """One decoded video frame, exactly as the decoder produced it.

    You cannot build one yourself: a :class:`VideoPacketDecoder` creates
    them. Use a :class:`ColorConverter` to turn them into RGB
    :class:`~torchcodec.Frame`\\ s, or you can read and transform the raw
    samples directly from :attr:`planes`::


        for packet in demuxer:
            for raw_frame in packet_decoder.decode(packet):
                y, u, v = raw_frame.planes
                # For a 480x270 yuv420p frame, y is [270, 480] uint8, and the
                # chroma is subsampled: u and v are [135, 240] each.
                print(y.shape, u.shape, v.shape)

    Nothing here has been converted. The samples are in the codec's own pixel
    format (typically YUV), on the device that was passed to
    :meth:`VideoStream.make_decoder`, and
    :attr:`width`, :attr:`height` and :attr:`planes` are all pre-rotation:
    :attr:`rotation` is what a :class:`ColorConverter` applies for you, and
    what you have to apply yourself if you convert :attr:`planes` on your own.

    .. important::

        On CUDA, the samples are produced on the CUDA stream that was current when
        you called :meth:`VideoPacketDecoder.decode`, and all of that work is
        enqueued by the time the call returns. If you consume them on any other stream
        (via :class:`ColorConverter` or by reading :attr:`planes`), you must
        handle synchronization yourself.  See
        :ref:`sphx_glr_generated_examples_low_level_cuda_streams.py`.
    """

    pts: int | None
    """The :term:`pts` of this frame, in units of the stream's time base, like
    the :attr:`Packet.pts` it comes from. ``None`` if the frame has no
    timestamp (e.g. in raw H.264 streams).

    Unlike :attr:`pts_seconds`, it is exact: use it to match a frame to the
    timestamp of a packet."""
    pts_seconds: float
    """The :term:`pts` of this frame, in seconds. ``NaN`` if the frame has no
    timestamp (e.g. in raw H.264 streams)."""
    duration_seconds: float
    """How long this frame is displayed for, in seconds."""
    storage_cuda: torch.Tensor | None
    """The CUDA allocation backing :attr:`planes`, or ``None`` on CPU.

    This tensor is exposed for one purpose, which is to let you call
    :meth:`torch.Tensor.record_stream` if you want to. See
    :ref:`sphx_glr_generated_examples_low_level_cuda_streams.py` for more details.
    You shouldn't read or write this tensor directly.
    """

    def __init__(
        self,
        handle: torch.Tensor,
        pts: int | None,
        pts_seconds: float,
        duration_seconds: float,
        storage: torch.Tensor | None = None,
    ):
        self._handle = handle
        self.storage_cuda = storage
        self.pts = pts
        self.pts_seconds = pts_seconds
        self.duration_seconds = duration_seconds
        self._metadata: _Metadata | None = None
        self._planes: tuple[torch.Tensor, ...] | None = None

    @property
    def _device(self) -> torch.device:
        return (
            self.storage_cuda.device
            if self.storage_cuda is not None
            else torch.device("cpu")
        )

    def _get_metadata(self) -> _Metadata:
        if self._metadata is None:
            self._metadata = _Metadata(*_blocks_frame_metadata(self._handle))
        return self._metadata

    @property
    def pixel_format(self) -> str:
        """The FFmpeg pixel-format name, e.g. ``"yuv420p"``.

        On CPU this is the source's own format. On CUDA it is always one of the
        NVDEC surface formats: ``"nv12"``, ``"p010le"``, ``"p012le"``,
        ``"p016le"``, ``"yuv444p"`` or ``"yuv444p16le"``.
        """
        return self._get_metadata().pixel_format

    @property
    def color_space(self) -> str:
        """The FFmpeg color space name, e.g. ``"bt709"``, or ``"unspecified"``.

        This describes :attr:`planes`, which is not always how the source is
        tagged: a CUDA decoder that falls back to the CPU converts an RGB frame
        into a YUV surface format, and this reports the color space of that
        conversion. Prefer it over
        :attr:`VideoStreamHeaderMetadata.color_space
        <torchcodec.decoders.VideoStreamMetadata.color_space>` when you convert
        the samples yourself.
        """
        return self._get_metadata().color_space

    @property
    def color_range(self) -> str:
        """``"tv"`` for limited range, ``"pc"`` for full range."""
        return self._get_metadata().color_range

    @property
    def color_primaries(self) -> str:
        """The FFmpeg color primaries name, e.g. ``"bt709"``, ``"bt2020"``, or
        ``"unspecified"``."""

        return self._get_metadata().color_primaries

    @property
    def color_transfer_characteristic(self) -> str:
        """The FFmpeg transfer characteristic name, e.g. ``"bt709"``,
        ``"smpte2084"`` (PQ), ``"arib-std-b67"`` (HLG), or ``"unspecified"``."""
        return self._get_metadata().color_transfer_characteristic

    @property
    def bit_depth(self) -> int:
        """How many bits of each :attr:`planes` sample are meaningful.

        In almost every case this is just the bit depth of the source: 8 for an
        8-bit video, 10 for a 10-bit one. It is worth having because a plane's
        dtype only tells you its storage width, ``uint8`` or ``uint16``, while
        this tells you the range of the values held in it. So it is what you
        shift or scale by to reach a range of your own -
        ``y >> (frame.bit_depth - 8)`` for 8 bits, or
        ``y / (2 ** frame.bit_depth - 1)`` to normalise.

        Two CUDA surface formats report more than their source: a 10-bit 4:4:4
        source is uploaded as ``yuv444p16le``, and a 12-bit source is tagged
        ``p016le`` on FFmpeg < 6, which has no ``p012le``. Both report 16 where
        CPU decoding would report 10 and 12. Their samples are msb-aligned, so
        they genuinely are 16-bit values with zeroed low bits, and the
        arithmetic above still holds.
        """
        return self._get_metadata().bit_depth

    @property
    def width(self) -> int:
        """The width of the decoded samples, before rotation."""
        return self._get_metadata().width

    @property
    def height(self) -> int:
        """The height of the decoded samples, before rotation."""
        return self._get_metadata().height

    @property
    def rotation(self) -> float:
        """How many degrees counter-clockwise the frame has to be rotated to be
        upright, or 0 if the container asks for no rotation.

        This is *not* applied to :attr:`planes`. A :class:`ColorConverter`
        applies it, rounded to the nearest multiple of 90, to its output.
        """
        return self._get_metadata().rotation

    @property
    def planes(self) -> tuple[torch.Tensor, ...]:
        """The decoder's own samples, as 2D tensor views.

        There is exactly one tensor per *component*
        of :attr:`pixel_format`, in the order that format describes, of dtype
        ``uint8`` or ``uint16`` depending on :attr:`bit_depth`. So ``yuv420p``
        and ``nv12`` both give three (``y, u, v = planes``), ``yuva420p`` four
        (``y, u, v, a = planes``) and ``gray`` one (``(y,) = planes``).

        They are always on the device that was passed to
        :meth:`VideoStream.make_decoder`, including when a CUDA decoder has to
        fall back to decoding on the CPU: it uploads those frames before handing
        them out.

        Only the luma and alpha components are :attr:`height` by :attr:`width`.
        The chroma ones are subsampled by whatever :attr:`pixel_format` says: half in
        both directions for a 4:2:0 format, half the width for 4:2:2, full size
        for 4:4:4 and for the RGB formats. Odd sizes round up, so the chroma of
        a 4:2:0 frame 481 samples wide is 241 wide.

        .. note::

            None of these views is contiguous in general. FFmpeg pads each row
            out to a line size of its own choosing, so even a luma plane is
            usually strided, and the semi-planar formats (``nv12``, ``p010le``,
            and the other NVDEC surface formats) store U and V interleaved in a
            single allocation, which forces those two to be strided views into
            it.

        Raises:
            RuntimeError: For the pixel formats that can't be viewed without a
                copy - sub-byte-packed, palettised and float ones - and for
                frames stored bottom-up. Check :attr:`pixel_format` first if you are
                decoding something exotic.
        """
        if self._planes is None:
            planes = _blocks_frame_planes(self._handle, self._device)
            # The op's schema is fixed-arity, so it pads with empty tensors up
            # to the 4 components of the widest format.
            self._planes = tuple(p for p in planes if p.numel() > 0)
        return self._planes


@dataclass
class RawAudioSamples:
    """One decoded audio frame's samples, exactly as the decoder produced them.

    You cannot build one yourself: an :class:`AudioPacketDecoder` creates them.
    Use an :class:`AudioConverter` to turn them into normalised float32
    :class:`~torchcodec.AudioSamples`, or read :attr:`data` directly::

        for packet in demuxer:
            for raw_samples in audio_packet_decoder.decode(packet):
                print(raw_samples.data.shape)   # e.g. [2, 1024]
                print(raw_samples.data.dtype)   # e.g. float32, for an fltp source
                print(raw_samples.sample_rate)  # e.g. 16000
    """

    data: torch.Tensor
    """Always a contiguous ``[num_channels, num_samples]`` tensor, whatever the
    source's sample format. Planar and packed sources alike come out with that
    same shape and layout (they are copied).

    The dtype is whichever one holds the source's samples exactly: ``uint8``
    for ``u8``, ``int16`` for ``s16``, ``int32`` for ``s32``, ``int64`` for
    ``s64``, ``float32`` for ``flt`` and ``float64`` for ``dbl``. The integer
    ones are *not* normalised to ``[-1, 1]``; that is what an
    :class:`AudioConverter` does."""
    sample_rate: int
    """The source's sample rate, in Hz."""
    pts: int | None
    """The :term:`pts` of the first sample, in units of the stream's time base,
    like the :attr:`Packet.pts` it comes from. ``None`` if it is unknown.

    Unlike :attr:`pts_seconds`, it is exact: use it to match samples to the
    timestamp of a packet."""
    pts_seconds: float
    """The :term:`pts` of the first sample, in seconds. ``NaN`` if it is
    unknown."""
    duration_seconds: float
    """How long these samples last, in seconds."""
    # See Packet._generation.
    _generation: int = 0

    @property
    def num_channels(self) -> int:
        """The number of channels, i.e. ``data.shape[0]``."""
        return self.data.shape[0]

    @property
    def num_samples(self) -> int:
        """The number of samples per channel, i.e. ``data.shape[1]``."""
        return self.data.shape[1]
