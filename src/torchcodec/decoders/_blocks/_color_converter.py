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
    """Turn a :class:`RawFrame` (typically YUV) into an RGB :class:`~torchcodec.Frame`.

    .. code-block:: python

        converter = ColorConverter()

        for packet in demuxer:
            for raw_frame in packet_decoder.decode(packet):
                frame = converter.convert(raw_frame)
                frame.data  # uint8 [3, height, width], RGB

    Unlike the other blocks this one isn't tied to a specific video stream.
    Everything it needs (dimensions, pixel format, color space, rotation) comes
    from the :class:`RawFrame` itself, so the same converter instance can
    process frames from any video stream, provided that they share the same
    device.

    Args:
        device (str or torch.device, optional): The device to convert on. If
            ``None`` (default), the current default device is used (see
            ``torch.set_default_device``). It has to be the device the frames
            are already on, i.e. it must match what was passed to the
            :class:`~torchcodec.decoders._blocks.VideoPacketDecoder` that produced
            the :class:`RawFrame`.
        output_dtype (torch.dtype or ``"auto"``, optional): ``torch.uint8``
            (default) for values in ``[0, 255]``, ``torch.float32`` for
            ``[0, 1]``, or ``"auto"`` for uint8 from 8-bit sources and float32
            from deeper ones. Since this block isn't tied to a stream,
            ``"auto"`` is resolved per frame rather than once per video, so
            feeding it a mix of SDR and HDR frames gives you a mix of dtypes.
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
        """Convert one :class:`RawFrame` to an RGB :class:`~torchcodec.Frame`.

        The stream's rotation is applied, so the output is upright and matches
        what a :class:`~torchcodec.decoders.VideoDecoder` gives you.

        Args:
            raw_frame (RawFrame): The frame to convert. It has to be on this
                converter's device.

        Returns:
            The RGB ``[3, height, width]`` frame in the converter's
            ``output_dtype``.

        Raises:
            RuntimeError: If the frame is not on this converter's device.
        """
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
