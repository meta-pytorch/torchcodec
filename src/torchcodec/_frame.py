# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from torch import Tensor


def _frame_repr(self):
    # Utility to replace __repr__ method of dataclasses below. This prints the
    # shape of the .data tensor rather than printing the (potentially very long)
    # data tensor itself.
    s = self.__class__.__name__ + ":\n"
    spaces = "  "
    for field in dataclasses.fields(self):
        field_name = field.name
        field_val = getattr(self, field_name)
        if field_name == "data":
            field_name = "data (shape)"
            field_val = field_val.shape
        s += f"{spaces}{field_name}: {field_val}\n"
    return s


@dataclass
class Frame(Iterable):
    """A single video frame with associated metadata."""

    data: Tensor
    """The frame data as (3-D ``torch.Tensor``)."""
    pts_seconds: float
    """The :term:`pts` of the frame, in seconds (float)."""
    duration_seconds: float
    """The duration of the frame, in seconds (float)."""

    def __post_init__(self):
        # This is called after __init__() when a Frame is created. We can run
        # input validation checks here.
        if not self.data.ndim == 3:
            raise ValueError(f"data must be 3-dimensional, got {self.data.shape = }")
        self.pts_seconds = float(self.pts_seconds)
        self.duration_seconds = float(self.duration_seconds)

    def __iter__(self) -> Iterator[Tensor | float]:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)

    def __repr__(self):
        return _frame_repr(self)


@dataclass
class FrameBatch(Iterable):
    """Multiple video frames with associated metadata.

    The ``data`` tensor is typically 4D for sequences of frames (NHWC or NCHW),
    or 5D for sequences of clips, as returned by the :ref:`samplers
    <sphx_glr_generated_examples_decoding_sampling.py>`. When ``data`` is 4D (resp.  5D)
    the ``pts_seconds`` and ``duration_seconds`` tensors are 1D (resp. 2D).

    .. note::
        The ``pts_seconds`` and ``duration_seconds`` Tensors are always returned
        on CPU, even if ``data`` is on GPU.
    """

    data: Tensor
    """The frames data (``torch.Tensor`` of uint8)."""
    pts_seconds: Tensor
    """The :term:`pts` of the frame, in seconds (``torch.Tensor`` of floats)."""
    duration_seconds: Tensor
    """The duration of the frame, in seconds (``torch.Tensor`` of floats)."""

    def __post_init__(self):
        # This is called after __init__() when a FrameBatch is created. We can
        # run input validation checks here.
        if self.data.ndim < 3:
            raise ValueError(
                f"data must be at least 3-dimensional, got {self.data.shape = }"
            )

        leading_dims = self.data.shape[:-3]
        if not (leading_dims == self.pts_seconds.shape == self.duration_seconds.shape):
            raise ValueError(
                "Tried to create a FrameBatch but the leading dimensions of the inputs do not match. "
                f"Got {self.data.shape = } so we expected the shape of pts_seconds and "
                f"duration_seconds to be {leading_dims = }, but got "
                f"{self.pts_seconds.shape = } and {self.duration_seconds.shape = }."
            )

    def __iter__(self) -> Iterator["FrameBatch"]:
        for data, pts_seconds, duration_seconds in zip(
            self.data, self.pts_seconds, self.duration_seconds
        ):
            yield FrameBatch(
                data=data,
                pts_seconds=pts_seconds,
                duration_seconds=duration_seconds,
            )

    def __getitem__(self, key) -> "FrameBatch":
        return FrameBatch(
            data=self.data[key],
            pts_seconds=self.pts_seconds[key],
            duration_seconds=self.duration_seconds[key],
        )

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return _frame_repr(self)


@dataclass
class MotionVectorBatch(Iterable):
    """Motion vectors and metadata for multiple video frames.

    The ``data`` tensor is padded to shape ``(N, max_mvs, 10)``. Each motion
    vector has 10 int32 fields in this order:
    1) source
    2) w
    3) h
    4) src_x
    5) src_y
    6) dst_x
    7) dst_y
    8) motion_x
    9) motion_y
    10) motion_scale

    The motion components follow the FFmpeg convention. In particular,
    ``dst_x`` and ``dst_y`` refer to the block position in the current frame,
    and ``src_x`` and ``src_y`` refer to the corresponding block position in
    the reference frame identified by ``source``. Coordinates can be
    sub-pixel, represented via the fixed-point fields ``motion_x``,
    ``motion_y``, and ``motion_scale``:
    ``src_x = dst_x + motion_x / motion_scale`` and
    ``src_y = dst_y + motion_y / motion_scale``.

    All tensors are on CPU. ``data``, ``counts``, and ``frame_types`` are int32.
    ``pts_seconds`` and ``duration_seconds`` are float64.

    .. note::

        If none of the requested frames contain motion vectors, ``max_mvs`` can
        be 0 and ``data`` will have shape ``(N, 0, 10)``.
    """

    data: Tensor
    """Motion vectors as a 3D tensor (N, max_mvs, 10) of int32."""
    counts: Tensor
    """Number of motion vectors per frame (1D tensor of int32)."""
    pts_seconds: Tensor
    """PTS for each frame in seconds (1D tensor of float64)."""
    duration_seconds: Tensor
    """Duration for each frame in seconds (1D tensor of float64)."""
    frame_types: Tensor
    """Frame types as ASCII codes (1D tensor of int32)."""

    def __post_init__(self):
        if self.data.ndim != 3:
            raise ValueError(f"data must be 3-dimensional, got {self.data.shape = }")
        leading_dim = self.data.shape[0]
        for name, tensor in (
            ("counts", self.counts),
            ("pts_seconds", self.pts_seconds),
            ("duration_seconds", self.duration_seconds),
            ("frame_types", self.frame_types),
        ):
            if tensor.ndim != 1 or tensor.shape[0] != leading_dim:
                raise ValueError(
                    f"{name} must be 1D with length {leading_dim}, got {tensor.shape = }."
                )

    def __iter__(self) -> Iterator["MotionVectorBatch"]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key) -> "MotionVectorBatch":
        data = self.data[key]
        counts = self.counts[key]
        pts_seconds = self.pts_seconds[key]
        duration_seconds = self.duration_seconds[key]
        frame_types = self.frame_types[key]

        if data.ndim == 2:
            data = data.unsqueeze(0)
        if counts.ndim == 0:
            counts = counts.unsqueeze(0)
            pts_seconds = pts_seconds.unsqueeze(0)
            duration_seconds = duration_seconds.unsqueeze(0)
            frame_types = frame_types.unsqueeze(0)

        return MotionVectorBatch(
            data=data,
            counts=counts,
            pts_seconds=pts_seconds,
            duration_seconds=duration_seconds,
            frame_types=frame_types,
        )

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return _frame_repr(self)


@dataclass
class AudioSamples(Iterable):
    """Audio samples with associated metadata."""

    data: Tensor
    """The sample data (``torch.Tensor`` of float in [-1, 1], shape is ``(num_channels, num_samples)``)."""
    pts_seconds: float
    """The :term:`pts` of the first sample, in seconds."""
    duration_seconds: float
    """The duration of the samples, in seconds."""
    sample_rate: int
    """The sample rate of the samples, in Hz."""

    def __post_init__(self):
        # This is called after __init__() when a Frame is created. We can run
        # input validation checks here.
        if not self.data.ndim == 2:
            raise ValueError(f"data must be 2-dimensional, got {self.data.shape = }")
        self.pts_seconds = float(self.pts_seconds)
        self.sample_rate = int(self.sample_rate)

    def __iter__(self) -> Iterator[Tensor | float]:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)

    def __repr__(self):
        return _frame_repr(self)
