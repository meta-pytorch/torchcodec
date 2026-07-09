# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Private, experimental building-block decode API (CPU only, for now).

Exposes the three decode stages -- :class:`Demuxer`, :class:`PacketDecoder`,
:class:`ColorConverter` -- as passive, composable, GIL-releasing units, so a
caller can build its own (threaded) decode pipeline and tune how the stages
overlap. The blocks do no threading themselves.

This is experimental and private; the API may change. See
API_breakdown_claude_plan.md for the design and rationale.
"""

from ._color_converter import ColorConverter
from ._demuxer import Demuxer
from ._frame import DecodedFrame, Packet
from ._packet_decoder import PacketDecoder

__all__ = ["Demuxer", "PacketDecoder", "ColorConverter", "Packet", "DecodedFrame"]
