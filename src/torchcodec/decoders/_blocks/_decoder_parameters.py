# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, KW_ONLY
from fractions import Fraction
from typing import Any

# TODO_DEMUXER_FREE: settle on the names of these two classes, and of
# VideoStream.decoder_parameters / AudioStream.decoder_parameters.

_COMMON_DOC = """
    The fields map one-to-one to FFmpeg's `AVCodecParameters
    <https://ffmpeg.org/doxygen/trunk/structAVCodecParameters.html>`_, plus
    :attr:`time_base`{extra}. Which ones a decoder needs depends on the codec:
    anything left to ``None`` keeps FFmpeg's default. To see what a given kind
    of stream needs, look at what a :class:`Demuxer` reports for a sample file:
    ``print(Demuxer("sample.mp4").streams[0].decoder_parameters)``.

    Leaving out a field the codec needs can make building the decoder fail, make
    decoding fail, or silently produce wrong frames or samples, depending on the
    codec.{example}

    Instances can be pickled, so they can be sent to other processes along with
    the :class:`Packet`\\ s they describe.

    Args:
        codec (str): The FFmpeg name of the codec, e.g. ``"h264"``. This is what
            the bitstream is, not which decoder implementation decodes it: that
            is chosen when building the decoder.
        time_base (Fraction or tuple of two ints): The unit of the stream's
            timestamps: a pts of ``n`` is ``n * time_base`` seconds.
"""

_VIDEO_EXAMPLE = """ Some common cases:

    ===========================================================  ===============================================
    Packets                                                      Fields to set besides ``codec`` and ``time_base``
    ===========================================================  ===============================================
    Annex B H.264 / HEVC, AV1, VP9, with in-band codec config    None
    Copied out of MP4, MKV, ...                                  :attr:`extradata` (e.g. the avcC / hvcC box)
    Raw video                                                    :attr:`width`, :attr:`height`, :attr:`pixel_format`
    Rotated video                                                :attr:`rotation`
    ===========================================================  ===============================================
"""

_AUDIO_EXAMPLE = """ Some common cases:

    ===========================  ==========================================================================
    Packets                      Fields to set besides ``codec`` and ``time_base``
    ===========================  ==========================================================================
    MP3, FLAC, AAC with ADTS     None
    AAC without ADTS headers     :attr:`extradata` (the AudioSpecificConfig)
    Opus                         :attr:`extradata` (the OpusHead) and :attr:`initial_padding`
    PCM                          :attr:`sample_format`, :attr:`sample_rate`, :attr:`channel_layout`
    ===========================  ==========================================================================
"""


def _to_fraction(name: str, value) -> Fraction:
    if isinstance(value, tuple):
        if len(value) != 2 or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        ):
            raise ValueError(f"{name} must be a Fraction or a tuple of two ints.")
        if value[1] == 0:
            raise ValueError(f"{name} can't have a denominator of 0.")
        return Fraction(*value)
    if isinstance(value, Fraction):
        return value
    raise ValueError(
        f"{name} must be a Fraction or a tuple of two ints, got {value!r}."
    )


# See decoder_parameters_to_strings() in DecoderParameters.h.
_CODED_SIDE_DATA_PREFIX = "coded_side_data:"


def _codec_tag_to_int(codec_tag: str | int) -> int:
    if isinstance(codec_tag, str):
        raw = codec_tag.encode("latin-1")
        if len(raw) != 4:
            raise ValueError(
                f"codec_tag must be a 4-character string or an int, got {codec_tag!r}."
            )
        # FFmpeg's MKTAG: the first character is the least significant byte.
        return int.from_bytes(raw, "little")
    return codec_tag


@dataclass(frozen=True)
class _DecoderParameters:
    codec: str
    # Passing a tuple is accepted, and normalized to a Fraction.
    time_base: Fraction | tuple[int, int]
    _: KW_ONLY
    extradata: bytes | None = None
    """Codec configuration that isn't carried by the packets themselves, e.g.
    the avcC box of an H.264 stream stored in MP4."""
    codec_tag: int | str | None = None
    """The FourCC of the codec, e.g. ``"avc1"``. Can be passed as a 4-character
    string, and is stored as an int."""
    profile: int | None = None
    """The codec's profile, as FFmpeg's ``AV_PROFILE_*`` value."""
    level: int | None = None
    bit_rate: int | None = None
    bits_per_coded_sample: int | None = None
    bits_per_raw_sample: int | None = None
    coded_side_data: dict[str, bytes] | None = None
    """Side data that applies to the whole stream, keyed by FFmpeg's side data
    name (e.g. ``"Content light level metadata"``). Requires FFmpeg 6.1 or
    above."""
    _packets_may_be_misaligned_after_reset: bool = field(
        default=False, init=False, repr=False, compare=False
    )

    _media_type = ""

    def __post_init__(self):
        if not isinstance(self.codec, str) or not self.codec:
            raise ValueError(f"codec must be a non-empty string, got {self.codec!r}.")
        time_base = _to_fraction("time_base", self.time_base)
        if time_base <= 0:
            raise ValueError(f"time_base must be positive, got {time_base}.")
        object.__setattr__(self, "time_base", time_base)
        if self.codec_tag is not None:
            object.__setattr__(self, "codec_tag", _codec_tag_to_int(self.codec_tag))
        for name in ("sample_aspect_ratio", "framerate", "r_frame_rate"):
            value = getattr(self, name, None)
            if value is not None:
                object.__setattr__(self, name, _to_fraction(name, value))

    def __repr__(self) -> str:
        fields = []
        for f in dataclasses.fields(self):
            if not f.repr:
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, bytes):
                value = f"<{len(value)} bytes>"
            elif isinstance(value, dict):
                value = (
                    "{"
                    + ", ".join(f"{k!r}: <{len(v)} bytes>" for k, v in value.items())
                    + "}"
                )
            elif isinstance(value, Fraction):
                value = str(value)
            else:
                value = repr(value)
            fields.append(f"{f.name}={value}")
        return f"{type(self).__name__}({', '.join(fields)})"

    @classmethod
    def _from_strings(cls, values: dict[str, str]):
        """From what the C++ decoder_parameters_to_strings() returns."""
        values = dict(values)
        misaligned = values.pop("packets_may_be_misaligned_after_reset") == "1"
        coded_side_data = {
            key[len(_CODED_SIDE_DATA_PREFIX) :]: bytes.fromhex(values.pop(key))
            for key in list(values)
            if key.startswith(_CODED_SIDE_DATA_PREFIX)
        }

        # Everything comes as a string: parse each by the type of its field.
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in values:
                continue
            value = values[f.name]
            kind = str(f.type).split(" ")[0]
            if kind == "int":
                kwargs[f.name] = int(value)
            elif kind == "float":
                kwargs[f.name] = float(value)
            elif kind == "Fraction":
                kwargs[f.name] = Fraction(value)
            elif kind == "bytes":
                kwargs[f.name] = bytes.fromhex(value)
            else:
                kwargs[f.name] = value

        parameters = cls(**kwargs, coded_side_data=coded_side_data or None)
        object.__setattr__(
            parameters, "_packets_may_be_misaligned_after_reset", misaligned
        )
        return parameters


@dataclass(frozen=True, repr=False, kw_only=True)
class VideoPacketDecoderParameters(_DecoderParameters):
    width: int | None = None
    height: int | None = None
    pixel_format: str | None = None
    """FFmpeg's name of the pixel format, e.g. ``"yuv420p"``."""
    sample_aspect_ratio: Fraction | tuple[int, int] | None = None
    """Can be passed as a tuple of two ints."""
    framerate: Fraction | tuple[int, int] | None = None
    """Can be passed as a tuple of two ints. Requires FFmpeg 6 or above."""
    r_frame_rate: Fraction | tuple[int, int] | None = None
    """FFmpeg's ``AVStream.r_frame_rate``: not a codec parameter, but the frame
    durations are derived from it when decoding on NVDEC, which doesn't report
    them. Can be passed as a tuple of two ints."""
    field_order: str | None = None
    """One of ``"unknown"``, ``"progressive"``, ``"tt"``, ``"bb"``, ``"tb"`` or
    ``"bt"``."""
    color_range: str | None = None
    """FFmpeg's name of the color range, ``"tv"`` or ``"pc"``."""
    color_primaries: str | None = None
    """FFmpeg's name of the color primaries, e.g. ``"bt709"``."""
    color_transfer_characteristic: str | None = None
    """FFmpeg's name of the transfer characteristic, e.g. ``"bt709"``."""
    color_space: str | None = None
    """FFmpeg's name of the color space, e.g. ``"bt709"``."""
    chroma_location: str | None = None
    """FFmpeg's name of the chroma location, e.g. ``"left"``."""
    video_delay: int | None = None
    rotation: float | None = None
    """How many degrees counter-clockwise the frames have to be rotated to be
    upright. Not a codec parameter: it comes from the container's display
    matrix, and is what :attr:`RawFrame.rotation` reports."""

    _media_type = "video"


@dataclass(frozen=True, repr=False, kw_only=True)
class AudioPacketDecoderParameters(_DecoderParameters):
    sample_format: str | None = None
    """FFmpeg's name of the sample format, e.g. ``"fltp"``."""
    sample_rate: int | None = None
    channel_layout: str | None = None
    """FFmpeg's description of the channel layout, e.g. ``"stereo"`` or
    ``"5.1"``."""
    block_align: int | None = None
    frame_size: int | None = None
    initial_padding: int | None = None
    trailing_padding: int | None = None
    seek_preroll: int | None = None

    _media_type = "audio"


VideoPacketDecoderParameters.__doc__ = """Describes the packets of a video stream, to build a
    :class:`VideoPacketDecoder` from.

    A :class:`VideoStream` provides them as
    :attr:`VideoStream.decoder_parameters`, but you can build one yourself to
    decode packets that don't come from a :class:`Demuxer`.
""" + _COMMON_DOC.format(
    extra=", :attr:`r_frame_rate` and :attr:`rotation`", example=_VIDEO_EXAMPLE
)
AudioPacketDecoderParameters.__doc__ = """Describes the packets of an audio stream, to build an
    :class:`AudioPacketDecoder` from.

    An :class:`AudioStream` provides them as
    :attr:`AudioStream.decoder_parameters`, but you can build one yourself to
    decode packets that don't come from a :class:`Demuxer`.
""" + _COMMON_DOC.format(
    extra="", example=_AUDIO_EXAMPLE
)
