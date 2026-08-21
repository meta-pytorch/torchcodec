# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Private, experimental building-block decode API.

Exposes the three decode stages -- :class:`VideoDemuxer`,
:class:`PacketDecoder`, :class:`ColorConverter` -- as passive, composable,
GIL-releasing units, so a caller can build its own (threaded) decode pipeline
and tune how the stages overlap. The blocks do no threading themselves.

:class:`AudioDemuxer` is the audio counterpart of :class:`VideoDemuxer`;
:class:`PacketDecoder` is shared, since decoding is the same operation either
way.

This is experimental and private; the API may change. See
API_breakdown_claude_plan.md for the design and rationale.
"""

from ._color_converter import ColorConverter
from ._demuxer import AudioDemuxer, StreamIndex, VideoDemuxer
from ._frame import Packet, RawAudioSamples, RawFrame
from ._packet_decoder import PacketDecoder

__all__ = [
    "VideoDemuxer",
    "AudioDemuxer",
    "PacketDecoder",
    "ColorConverter",
    "Packet",
    "RawFrame",
    "RawAudioSamples",
    "StreamIndex",
]

# TODO_API_BREAKDOWN FEAT P1 we probably need a way to expose the *header*
# metadata - something that would avoid using VideoDecoder? VideoDemuxer.scan()
# covers the content-derived half of that. Audio makes this more pressing:
# there is no scan() there, so sample_rate / num_channels / sample_format are
# only reachable through AudioDecoder today.
