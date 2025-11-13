# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType
from typing import Sequence

from torch import nn


@dataclass
class DecoderTransform(ABC):
    """Base class for all decoder transforms.

    A *decoder transform* is a transform that is applied by the decoder before
    returning the decoded frame.  Applying decoder transforms to frames
    should be both faster and more memory efficient than receiving normally
    decoded frames and applying the same kind of transform.

    Most ``DecoderTransform`` objects have a complementary transform in TorchVision,
    specificially in `torchvision.transforms.v2 <https://docs.pytorch.org/vision/stable/transforms.html>`_. For such transforms, we
    ensure that:

      1. The names are the same.
      2. Default behaviors are the same.
      3. The parameters for the ``DecoderTransform`` object are a subset of the
         TorchVision :class:`~torchvision.transforms.v2.Transform` object.
      4. Parameters with the same name control the same behavior and accept a
         subset of the same types.
      5. The difference between the frames returned by a decoder transform and
         the complementary TorchVision transform are such that a model should
         not be able to tell the difference.
    """

    @abstractmethod
    def _make_transform_spec(self) -> str:
        pass


def import_torchvision_transforms_v2() -> ModuleType:
    try:
        from torchvision.transforms import v2
    except ImportError as e:
        raise RuntimeError(
            "Cannot import TorchVision; this should never happen, please report a bug."
        ) from e
    return v2


@dataclass
class Resize(DecoderTransform):
    """Resize the decoded frame to a given size.

    Complementary TorchVision transform: :class:`~torchvision.transforms.v2.Resize`.
    Interpolation is always bilinear. Anti-aliasing is always on.

    Args:
        size: (sequence of int): Desired output size. Must be a sequence of
            the form (height, width).
    """

    size: Sequence[int]

    def _make_transform_spec(self) -> str:
        assert len(self.size) == 2
        return f"resize, {self.size[0]}, {self.size[1]}"

    @classmethod
    def _from_torchvision(cls, resize_tv: nn.Module):
        v2 = import_torchvision_transforms_v2()

        assert isinstance(resize_tv, v2.Resize)

        if resize_tv.interpolation is not v2.InterpolationMode.BILINEAR:
            raise ValueError(
                "TorchVision Resize transform must use bilinear interpolation."
            )
        if resize_tv.antialias is False:
            raise ValueError(
                "TorchVision Resize transform must have antialias enabled."
            )
        if resize_tv.size is None:
            raise ValueError("TorchVision Resize transform must have a size specified.")
        if len(resize_tv.size) != 2:
            raise ValueError(
                "TorchVision Resize transform must have a (height, width) "
                f"pair for the size, got {resize_tv.size}."
            )
        return cls(size=resize_tv.size)
