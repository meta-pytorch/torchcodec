# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, KW_ONLY
from fractions import Fraction


def _check_fraction(name: str, value: object) -> None:
    if not isinstance(value, Fraction):
        raise ValueError(f"{name} must be a Fraction, got {value!r}.")


def _short_repr(value: object) -> str:
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        entries = ", ".join(f"{k!r}: {_short_repr(v)}" for k, v in value.items())
        return "{" + entries + "}"
    return repr(value)


@dataclass(frozen=True, repr=False)
class _CodecParameters:
    codec: str
    """FFmpeg's name of the codec, e.g. ``"h264"``, ``"hevc"`` or ``"aac"``.
    ``ffmpeg -codecs`` lists them."""
    _: KW_ONLY
    time_base: Fraction
    """The unit of the packets' timestamps, in seconds: a packet's ``pts`` of
    ``n`` means ``n * time_base`` seconds."""
    extradata: bytes | None = None
    """The codec's setup data that isn't in the packets, e.g. the ``avcC`` box
    of H.264 in MP4. Not all codecs and streams need it."""
    codec_tag: int | None = None
    """The container's identifier of the codec, e.g. a FourCC in AVI or MP4.
    A few codecs read it to tell variants apart."""
    profile: int | None = None
    """The codec's profile, as one of FFmpeg's ``AV_PROFILE_*`` values."""
    level: int | None = None
    """The codec's level."""
    bit_rate: int | None = None
    """The average bit rate, in bits per second."""
    bits_per_coded_sample: int | None = None
    """The number of bits per sample in the compressed data, for the codecs
    that need it (mostly uncompressed and PCM codecs)."""
    bits_per_raw_sample: int | None = None
    """The number of bits per sample in the decoded data, when it's not implied
    by the pixel or sample format, e.g. 10 for 10-bit video."""
    coded_side_data: dict[str, bytes] | None = None
    """Extra data about the stream, keyed by FFmpeg's names for its types, e.g.
    ``"Display Matrix"`` for video or ``"Replay Gain"`` for audio.

    If a video stream has a display matrix, it is under ``"Display Matrix"``
    on every FFmpeg version. FFmpeg only moved the stream's side data into
    ``AVCodecParameters`` in 6.1: before that, the display matrix comes from
    the stream's own side data (``AVStream.side_data``), and it is the only
    entry supported, so audio streams have none."""
    # FFmpeg's name of the format of the container the stream was demuxed from,
    # e.g. "mpeg", or None when the parameters don't come from a Demuxer.
    # Knowing the container lets the decoder keep behavior that is more accurate
    # than what it can do without it: resyncing after a seek in an MPEG program
    # stream, and picking NVDEC's bitstream filter. Without the container, the
    # first packets after a seek in an MPEG program stream raise instead of
    # being skipped (see
    # test_mpeg_program_stream_resync_needs_the_container_format), which we
    # expect to be rare. For NVDEC, we expect no difference in practice, but we
    # keep the container to be safe. It survives pickling, but it's not part of
    # the API.
    _container_format_PRIVATE_DO_NOT_USE: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        if not isinstance(self.codec, str) or not self.codec:
            raise ValueError(f"codec must be a non-empty string, got {self.codec!r}.")
        _check_fraction("time_base", self.time_base)
        if self.time_base <= 0:
            raise ValueError(f"time_base must be positive, got {self.time_base}.")

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{f.name}={_short_repr(getattr(self, f.name))}"
            for f in dataclasses.fields(self)
            if f.repr and getattr(self, f.name) is not None
        )
        return f"{type(self).__name__}({fields})"


@dataclass(frozen=True, repr=False, kw_only=True)
class VideoCodecParameters(_CodecParameters):
    """What a :class:`VideoPacketDecoder` needs to know to decode the packets
    of a video stream.

    If you have a :class:`Demuxer`, :attr:`VideoStream.codec_parameters` gives
    you the parameters of a stream. They can be pickled, e.g. to build the
    decoder in another process. To decode packets that don't come from a
    ``Demuxer``, build them yourself::

        codec_parameters = VideoCodecParameters(
            "h264", time_base=Fraction(1, 90_000), extradata=extradata
        )
        decoder = VideoPacketDecoder(codec_parameters)

    The fields mirror those of FFmpeg's `AVCodecParameters
    <https://ffmpeg.org/doxygen/trunk/structAVCodecParameters.html>`_, with the
    same names and meanings. FFmpeg's names stand for its enums, e.g.
    ``pixel_format="yuv420p"``. A field left to ``None`` keeps FFmpeg's
    default. ``codec`` is the only positional argument.

    Which fields you need depends on the codec: printing the parameters a
    ``Demuxer`` reports for a similar file shows what FFmpeg uses. **A missing
    field doesn't always raise**: depending on the codec, it can give you wrong
    frames instead.

    The parameters are meant to be passed to a decoder or stored, not read as
    metadata: use :attr:`VideoStream.metadata` for that. The values a
    ``Demuxer`` reports are FFmpeg's, and can change across FFmpeg and
    TorchCodec versions. The fields' names and meanings are stable.

    The frames' rotation comes from the ``"Display Matrix"`` entry of
    :attr:`coded_side_data`: the 3x3 matrix, as FFmpeg stores it.
    """

    width: int | None = None
    """The width of the frames, in pixels."""
    height: int | None = None
    """The height of the frames, in pixels."""
    pixel_format: str | None = None
    """FFmpeg's name of the decoded frames' pixel format, e.g. ``"yuv420p"``.
    ``ffmpeg -pix_fmts`` lists them."""
    sample_aspect_ratio: Fraction | None = None
    """The width of a pixel divided by its height."""
    framerate: Fraction | None = None
    """The frame rate, in frames per second, as the codec reports it. Setting
    it requires FFmpeg 6.1 or above."""
    r_frame_rate: Fraction | None = None
    """The frame rate, in frames per second, as the container reports it. Not
    part of FFmpeg's ``AVCodecParameters``, but of its ``AVStream``."""
    field_order: str | None = None
    """``"progressive"`` or, for interlaced video, which field comes first:
    ``"tt"``, ``"bb"``, ``"tb"`` or ``"bt"``."""
    color_range: str | None = None
    """FFmpeg's name of the color range, e.g. ``"tv"`` (limited) or ``"pc"``
    (full)."""
    color_primaries: str | None = None
    """FFmpeg's name of the color primaries, e.g. ``"bt709"``."""
    color_transfer_characteristic: str | None = None
    """FFmpeg's name of the transfer characteristic, e.g. ``"bt709"`` or
    ``"smpte2084"``."""
    color_space: str | None = None
    """FFmpeg's name of the color space, i.e. the YUV matrix, e.g.
    ``"bt709"``."""
    chroma_location: str | None = None
    """FFmpeg's name of the chroma samples' location, e.g. ``"left"``."""
    video_delay: int | None = None
    """The number of frames the decoder holds back before it outputs one,
    because of B-frames."""

    def __post_init__(self):
        super().__post_init__()
        for name in ("sample_aspect_ratio", "framerate", "r_frame_rate"):
            value = getattr(self, name)
            if value is not None:
                _check_fraction(name, value)


@dataclass(frozen=True, repr=False, kw_only=True)
class AudioCodecParameters(_CodecParameters):
    """What an :class:`AudioPacketDecoder` needs to know to decode the packets
    of an audio stream.

    If you have a :class:`Demuxer`, :attr:`AudioStream.codec_parameters` gives
    you the parameters of a stream. They can be pickled, e.g. to build the
    decoder in another process. To decode packets that don't come from a
    ``Demuxer``, build them yourself::

        codec_parameters = AudioCodecParameters(
            "aac",
            time_base=Fraction(1, 48_000),
            sample_rate=48_000,
            channel_layout="stereo",
            extradata=extradata,
        )
        decoder = AudioPacketDecoder(codec_parameters)

    The fields mirror those of FFmpeg's `AVCodecParameters
    <https://ffmpeg.org/doxygen/trunk/structAVCodecParameters.html>`_, with the
    same names and meanings. FFmpeg's names stand for its enums, e.g.
    ``sample_format="fltp"``. A field left to ``None`` keeps FFmpeg's default.
    ``codec`` is the only positional argument.

    Which fields you need depends on the codec: printing the parameters a
    ``Demuxer`` reports for a similar file shows what FFmpeg uses. **A missing
    field doesn't always raise**: depending on the codec, it can give you wrong
    samples instead.

    The parameters are meant to be passed to a decoder or stored, not read as
    metadata: use :attr:`AudioStream.metadata` for that. The values a
    ``Demuxer`` reports are FFmpeg's, and can change across FFmpeg and
    TorchCodec versions. The fields' names and meanings are stable.
    """

    sample_format: str | None = None
    """FFmpeg's name of the decoded samples' format, e.g. ``"fltp"``.
    ``ffmpeg -sample_fmts`` lists them."""
    sample_rate: int | None = None
    """The sample rate, in Hz."""
    channel_layout: str | None = None
    """FFmpeg's description of the channel layout, e.g. ``"stereo"`` or
    ``"5.1"``. ``ffmpeg -layouts`` lists them."""
    block_align: int | None = None
    """The number of bytes per coded audio frame, for the codecs that need it
    (e.g. PCM and ADPCM codecs)."""
    frame_size: int | None = None
    """The number of samples per channel in a frame, for the codecs whose
    frames all have the same size."""
    initial_padding: int | None = None
    """The number of samples the encoder added at the start of the audio,
    which aren't part of the original audio."""
    trailing_padding: int | None = None
    """The number of samples the encoder added at the end of the audio, which
    aren't part of the original audio."""
    seek_preroll: int | None = None
    """The number of samples to skip after a discontinuity, e.g. a seek."""
