# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchcodec._core.ops import _blocks_convert_frame, _blocks_create_color_converter
from torchcodec._frame import Frame

from ._frame import DecodedFrame

# TODO_API_BREAKDOWN support output_dtype?
# TODO_API_BREAKDOWN Expose output_shape?

# TODO_API_BREAKDOWN We need to support rotation metadata!!


class ColorConverter:
    """Color-conversion building block: turns a decoded (YUV)
    :class:`DecodedFrame` into an RGB :class:`~torchcodec._frame.Frame`
    (CHW, uint8 -- matching ``VideoDecoder``'s default output).

    Self-contained and not bound to anything: everything it needs (dims, pixel
    format, colorspace, and on CUDA the hardware context) comes from the frame
    itself, so one converter can process frames from any video -- it needs no
    reference to the decoder that produced them. ``device`` selects where the
    conversion runs; on CUDA use ``device_variant="ffmpeg"`` to match a
    ``PacketDecoder`` configured the same way.

    Passive and *not* thread-safe: use one ``ColorConverter`` per thread.

    Note: automatic rotation (from stream side data) is not applied, since this
    block is intentionally stream-agnostic.
    """

    def __init__(self, device: str = "cpu", device_variant: str = "default"):
        self._handle = _blocks_create_color_converter(device, device_variant)

    def convert(self, decoded_frame: DecodedFrame) -> Frame:
        data = _blocks_convert_frame(self._handle, decoded_frame._handle)
        # The core op produces HWC; permute to CHW to match VideoDecoder (which
        # also returns a non-contiguous permuted view).
        data = data.permute(2, 0, 1)
        return Frame(
            data=data,
            pts_seconds=decoded_frame.pts_seconds,
            duration_seconds=decoded_frame.duration_seconds,
        )
