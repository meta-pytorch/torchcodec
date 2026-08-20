# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Literal

import torch

from torchcodec._core.ops import _blocks_convert_frame, _blocks_create_color_converter
from torchcodec._frame import Frame

from .._decoder_utils import convert_device_to_str, convert_output_dtype_to_str
from ._frame import DecodedFrame


class ColorConverter:
    """Color-conversion building block: turns a decoded (YUV)
    :class:`DecodedFrame` into an RGB :class:`~torchcodec._frame.Frame` (CHW).

    Not bound to anything: everything it needs (dims, pixel format, colorspace)
    comes from the frame itself, so one converter can process frames from any
    video. Passive and *not* thread-safe: use one ``ColorConverter`` per thread.

    ``output_dtype`` takes the same values as ``VideoDecoder``'s:
    ``torch.uint8`` (default, ``[0, 255]``), ``torch.float32`` (``[0, 1]``), or
    ``"auto"`` (uint8 for 8-bit sources, float32 for higher bit depths).
    Because this block is unbound, ``"auto"`` is resolved per frame rather than
    once per stream, so feeding it a mix of SDR and HDR frames yields a mix of
    dtypes.

    Rotation is applied too, so the output matches ``VideoDecoder``'s. The angle
    is part of the frame, like its dims and colorspace, so honoring it doesn't
    bind the converter to a stream either.

    ``device`` accepts a string or a ``torch.device``. It defaults to ``None``,
    which means the current default device (see ``torch.set_default_device``).
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
    ):
        self._handle = _blocks_create_color_converter(
            device=convert_device_to_str(device),
            output_dtype=convert_output_dtype_to_str(output_dtype),
        )

    def convert(self, decoded_frame: DecodedFrame) -> Frame:
        data = _blocks_convert_frame(self._handle, decoded_frame._handle)
        if decoded_frame.storage is not None:
            # See [Standalone Frame Storage and the need for record_stream]
            decoded_frame.storage.record_stream(torch.cuda.current_stream())
        # The core op produces HWC; permute to CHW to match VideoDecoder (which
        # also returns a non-contiguous permuted view).
        data = data.permute(2, 0, 1)
        return Frame(
            data=data,
            pts_seconds=decoded_frame.pts_seconds,
            duration_seconds=decoded_frame.duration_seconds,
        )
