# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Private, experimental building-block decode API.

Exposes the three decode stages -- :class:`VideoDemuxer`,
:class:`VideoPacketDecoder`, :class:`ColorConverter` -- as passive, composable,
GIL-releasing units, so a caller can build its own (threaded) decode pipeline
and tune how the stages overlap. The blocks do no threading themselves.

This is experimental and private; the API may change. See
API_breakdown_claude_plan.md for the design and rationale.
"""

from ._audio_converter import AudioConverter
from ._color_converter import ColorConverter
from ._demuxer import (
    AudioDemuxer,
    # TODO_API_BREAKDOWN DESIGN P1: AudioStream and VideoStream may conflict
    # with the encoder-side classes of the same name. Maybe that's OK?
    AudioStream,
    Demuxer,
    FrameIndex,
    get_container_metadata,
    VideoDemuxer,
    VideoStream,
)
from ._frame import Packet, RawAudioSamples, RawFrame
from ._packet_decoder import AudioPacketDecoder, VideoPacketDecoder

__all__ = [
    "Demuxer",
    "get_container_metadata",
    "VideoStream",
    "AudioStream",
    "VideoDemuxer",
    "AudioDemuxer",
    "VideoPacketDecoder",
    "AudioPacketDecoder",
    "ColorConverter",
    "AudioConverter",
    "Packet",
    "RawFrame",
    "RawAudioSamples",
    "FrameIndex",
]
