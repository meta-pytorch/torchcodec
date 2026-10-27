# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import NamedTuple

import torch

from torchcodec._core.ops import (
    _blocks_create_packet,
    _blocks_frame_metadata,
    _blocks_frame_planes,
    _blocks_packet_state,
    _blocks_packet_timing,
)

from ._helpers import _process_local


# FFmpeg's AV_NOPTS_VALUE and AV_PKT_FLAG_KEY.
_NO_PTS = -(2**63)
_KEY_FRAME_FLAG = 1


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


class Packet:
    """One compressed packet of one stream.

    A :class:`Demuxer` produces them, and a :class:`VideoPacketDecoder` or an
    :class:`AudioPacketDecoder` consumes them. To decode compressed data that
    doesn't come from a ``Demuxer``, build them with :meth:`from_bytes` or
    :meth:`from_tensor`.

    The timestamps of a packet (:attr:`pts`, :attr:`dts`, :attr:`duration`) are
    integers in the ``time_base`` of the decoder parameters of its stream.

    Packets can be pickled, e.g. to decode them in another process: send them
    along with the :attr:`VideoStream.decoder_parameters` or
    :attr:`AudioStream.decoder_parameters` of their stream, and build a decoder
    from those on the other side. Pickled packets can only be loaded by the
    same version of TorchCodec.
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
        is_key_frame: bool = False,
        side_data: dict[str, bytes] | None = None,
        stream_index: int = 0,
    ) -> Packet:
        """Build a packet from compressed data, e.g. one you received from the
        network or read from a database.

        ``data`` is copied. Only the data itself is needed to decode most codecs:
        the other fields are carried through to the decoded frames, or describe
        the packet. A decoder expects the packets of its stream in decoding
        order, one access unit (e.g. one encoded picture) per packet.

        Args:
            data (bytes, bytearray or memoryview): The compressed data.
            pts (int, optional): The presentation timestamp, in the time base of
                the stream's decoder parameters. It is what the
                :term:`pts` of the decoded frame is derived from.
            dts (int, optional): The decoding timestamp, in the same time base.
                Not needed for decoding.
            duration (int, optional): How long the packet lasts, in the same
                time base. 0 means unknown. Default: 0.
            is_key_frame (bool, optional): Whether the packet holds a key frame.
                Most codecs find that in the data itself. Default: False.
            side_data (dict, optional): Side data, keyed by FFmpeg's side data
                name, e.g. ``"Skip Samples"``.
            stream_index (int, optional): The value of :attr:`stream_index`.
                Default: 0.

        Returns:
            Packet: The packet.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"data must be bytes, bytearray or memoryview, got {type(data).__name__}."
            )
        if len(data) == 0:
            raise ValueError("data must not be empty.")
        with warnings.catch_warnings():
            # torch warns that the buffer of a bytes object isn't writable. The
            # op only reads it.
            warnings.filterwarnings("ignore", category=UserWarning)
            tensor = torch.frombuffer(data, dtype=torch.uint8)
        return cls._create(
            tensor,
            pts=pts,
            dts=dts,
            duration=duration,
            is_key_frame=is_key_frame,
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
        is_key_frame: bool = False,
        side_data: dict[str, bytes] | None = None,
        stream_index: int = 0,
    ) -> Packet:
        """Build a packet from compressed data held in a tensor.

        Same as :meth:`from_bytes`, but ``data`` is a 1D ``uint8`` CPU tensor.
        It is copied.
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"data must be a tensor, got {type(data).__name__}.")
        if data.dtype != torch.uint8 or data.ndim != 1 or data.device.type != "cpu":
            raise ValueError(
                "data must be a 1D uint8 CPU tensor, got a "
                f"{data.ndim}D {data.dtype} tensor on {data.device}."
            )
        if data.numel() == 0:
            raise ValueError("data must not be empty.")
        return cls._create(
            data.contiguous(),
            pts=pts,
            dts=dts,
            duration=duration,
            is_key_frame=is_key_frame,
            side_data=side_data,
            stream_index=stream_index,
        )

    @classmethod
    def _create(
        cls,
        data: torch.Tensor,
        *,
        pts: int | None,
        dts: int | None,
        duration: int,
        is_key_frame: bool,
        side_data: dict[str, bytes] | None,
        stream_index: int,
    ) -> Packet:
        side_data_layout = []
        payload = bytearray()
        for name, value in (side_data or {}).items():
            side_data_layout += [name, str(len(value))]
            payload += value
        handle = _blocks_create_packet(
            data,
            _NO_PTS if pts is None else pts,
            _NO_PTS if dts is None else dts,
            duration,
            _KEY_FRAME_FLAG if is_key_frame else 0,
            side_data_layout,
            (
                torch.frombuffer(payload, dtype=torch.uint8)
                if payload
                else torch.empty(0, dtype=torch.uint8, device="cpu")
            ),
        )
        return cls(handle, stream_index)

    @property
    def pts(self) -> int | None:
        """The presentation timestamp, or ``None`` if the packet has none."""
        pts = _blocks_packet_timing(self._handle)[0]
        return None if pts == _NO_PTS else pts

    @property
    def dts(self) -> int | None:
        """The decoding timestamp, or ``None`` if the packet has none."""
        dts = _blocks_packet_timing(self._handle)[1]
        return None if dts == _NO_PTS else dts

    @property
    def duration(self) -> int:
        """How long the packet lasts. 0 means unknown."""
        return _blocks_packet_timing(self._handle)[2]

    @property
    def is_key_frame(self) -> bool:
        """Whether the packet holds a key frame."""
        return bool(_blocks_packet_timing(self._handle)[3] & _KEY_FRAME_FLAG)

    @property
    def data(self) -> torch.Tensor:
        """A copy of the compressed data, as a 1D ``uint8`` tensor."""
        return _blocks_packet_state(self._handle)[0]

    @property
    def side_data(self) -> dict[str, bytes]:
        """The packet's side data, keyed by FFmpeg's side data name."""
        _, _, _, _, _, side_data_json, payload = _blocks_packet_state(self._handle)
        payload_bytes = payload.numpy().tobytes()
        side_data = {}
        offset = 0
        for name, size in json.loads(side_data_json):
            side_data[name] = payload_bytes[offset : offset + size]
            offset += size
        return side_data

    def __reduce__(self):
        data, pts, dts, duration, flags, side_data_json, side_data_payload = (
            _blocks_packet_state(self._handle)
        )
        side_data = []
        for name, size in json.loads(side_data_json):
            side_data += [name, str(size)]
        return (
            _rebuild_packet,
            (
                data,
                pts,
                dts,
                duration,
                flags,
                side_data,
                side_data_payload,
                self.stream_index,
                self._generation,
            ),
        )


def _rebuild_packet(
    data: torch.Tensor,
    pts: int,
    dts: int,
    duration: int,
    flags: int,
    side_data: list[str],
    side_data_payload: torch.Tensor,
    stream_index: int,
    generation: int,
) -> Packet:
    handle = _blocks_create_packet(
        data, pts, dts, duration, flags, side_data, side_data_payload
    )
    return Packet(handle, stream_index, generation=generation)


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
    """The :term:`pts` of this frame, exactly as the codec reported it: an
    integer in the decoder's :attr:`~VideoPacketDecoder.time_base`, or ``None``
    if the frame has no timestamp (e.g. raw H.264 streams)."""
    pts_seconds: float
    """The :term:`pts` of this frame, in seconds. ``NaN`` when :attr:`pts` is
    ``None``."""
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
    """The :term:`pts` of the first sample: an integer in the decoder's
    :attr:`~AudioPacketDecoder.time_base`, or ``None`` if it is unknown."""
    pts_seconds: float
    """The :term:`pts` of the first sample, in seconds. ``NaN`` when
    :attr:`pts` is ``None``."""
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
