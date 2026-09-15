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
from ._frame import RawFrame


class ColorConverter:
    def __init__(
        self,
        device: str | torch.device | None = None,
        output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
    ):
        self._handle = _blocks_create_color_converter(
            device=convert_device_to_str(device),
            output_dtype=convert_output_dtype_to_str(output_dtype),
        )

    # TODO_API_BREAKDOWN DESIGN P2: The frame device must match the converter
    # device. We have two alternative options:
    # - not take a device parameter in the constructor and make the converter
    #   device-agnostic. It requires caching the interfaces on the Converter.
    # - take a device parameter and always honor it: that means downloading or
    #   uploading CPU frames when needed.
    # I feel like maybe we want to allow a CPU frame to be CCed on the GPU and
    # do the upload ourselves (the download makes no sense, it's super slow).
    # Anyway, that can be done later.
    def convert(self, raw_frame: RawFrame) -> Frame:
        data = _blocks_convert_frame(self._handle, raw_frame._handle, raw_frame._device)
        if raw_frame._device.type == "cuda":
            # See [Standalone Frame Storage and the need for record_stream]
            raw_frame.record_stream(torch.cuda.current_stream())
        # The core op produces HWC; permute to CHW to match VideoDecoder (which
        # also returns a non-contiguous permuted view).
        data = data.permute(2, 0, 1)
        return Frame(
            data=data,
            pts_seconds=raw_frame.pts_seconds,
            duration_seconds=raw_frame.duration_seconds,
        )
