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
    _: KW_ONLY
    time_base: Fraction
    extradata: bytes | None = None
    codec_tag: int | None = None
    profile: int | None = None
    level: int | None = None
    bit_rate: int | None = None
    bits_per_coded_sample: int | None = None
    bits_per_raw_sample: int | None = None
    coded_side_data: dict[str, bytes] | None = None
    # FFmpeg's name of the format of the container the stream was demuxed from,
    # e.g. "mpeg", or None when the parameters don't come from a Demuxer.
    # Knowing the container lets the decoder keep behavior that is more accurate
    # than what it can do without it: resyncing after a seek in an MPEG program
    # stream, and picking NVDEC's bitstream filter. The difference is expected
    # to be rare and minor for MPEG program streams, and nonexistent in practice
    # for NVDEC once its filter is picked from the extradata when the container
    # is unknown, but we keep the container to be safe. It survives pickling,
    # but it's not part of the API.
    # TODO_DEMUXER_FREE: pick NVDEC's filter from the extradata without a
    # container. Until then, NVDEC raises without one.
    _container_format: str | None = field(
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


# TODO_DEMUXER_FREE: add docs. They should say that:
# 1. This mirrors FFmpeg's AVCodecParameters, and is meant to be passed to a
#    decoder or stored, not read as metadata: stream properties are read from
#    stream.metadata.
# 2. The values a Demuxer reports are whatever FFmpeg reports, and can change
#    between FFmpeg or TorchCodec versions.
# 3. The constructor's field names and meanings are stable public API.
# 4. The frames' rotation comes from the "Display Matrix" entry of
#    coded_side_data, the raw 3x3 matrix as FFmpeg stores it: there is no
#    separate rotation field. On FFmpeg < 6.1 it's the only coded_side_data
#    entry there can be, since FFmpeg only moved the stream's side data into
#    AVCodecParameters in 6.1.
@dataclass(frozen=True, repr=False, kw_only=True)
class VideoCodecParameters(_CodecParameters):
    width: int | None = None
    height: int | None = None
    pixel_format: str | None = None
    sample_aspect_ratio: Fraction | None = None
    framerate: Fraction | None = None
    r_frame_rate: Fraction | None = None
    field_order: str | None = None
    color_range: str | None = None
    color_primaries: str | None = None
    color_transfer_characteristic: str | None = None
    color_space: str | None = None
    chroma_location: str | None = None
    video_delay: int | None = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("sample_aspect_ratio", "framerate", "r_frame_rate"):
            value = getattr(self, name)
            if value is not None:
                _check_fraction(name, value)


# TODO_DEMUXER_FREE: add docs, making the same points as for
# VideoCodecParameters.
@dataclass(frozen=True, repr=False, kw_only=True)
class AudioCodecParameters(_CodecParameters):
    sample_format: str | None = None
    sample_rate: int | None = None
    channel_layout: str | None = None
    block_align: int | None = None
    frame_size: int | None = None
    initial_padding: int | None = None
    trailing_padding: int | None = None
    seek_preroll: int | None = None
