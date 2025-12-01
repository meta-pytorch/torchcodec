# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType
from typing import Optional, Sequence, Tuple

import torch
from torch import nn


@dataclass
class DecoderTransform(ABC):
    """Base class for all decoder transforms.

    A *decoder transform* is a transform that is applied by the decoder before
    returning the decoded frame.  Applying decoder transforms to frames
    should be both faster and more memory efficient than receiving normally
    decoded frames and applying the same kind of transform.

    Most ``DecoderTransform`` objects have a complementary transform in TorchVision,
    specificially in `torchvision.transforms.v2 <https://docs.pytorch.org/vision/stable/transforms.html>`_.
    For such transforms, we ensure that:

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
    def _make_transform_spec(
        self, input_dims: Tuple[Optional[int], Optional[int]]
    ) -> str:
        pass

    def _calculate_output_dims(
        self, input_dims: Tuple[Optional[int], Optional[int]]
    ) -> Tuple[Optional[int], Optional[int]]:
        return input_dims


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

    def _make_transform_spec(
        self, input_dims: Tuple[Optional[int], Optional[int]]
    ) -> str:
        # TODO: establish this invariant in the constructor during refactor
        assert len(self.size) == 2
        return f"resize, {self.size[0]}, {self.size[1]}"

    def _calculate_output_dims(
        self, input_dims: Tuple[Optional[int], Optional[int]]
    ) -> Tuple[Optional[int], Optional[int]]:
        # TODO: establish this invariant in the constructor during refactor
        assert len(self.size) == 2
        return (self.size[0], self.size[1])

    @classmethod
    def _from_torchvision(cls, tv_resize: nn.Module):
        v2 = import_torchvision_transforms_v2()

        assert isinstance(tv_resize, v2.Resize)

        if tv_resize.interpolation is not v2.InterpolationMode.BILINEAR:
            raise ValueError(
                "TorchVision Resize transform must use bilinear interpolation."
            )
        if tv_resize.antialias is False:
            raise ValueError(
                "TorchVision Resize transform must have antialias enabled."
            )
        if tv_resize.size is None:
            raise ValueError("TorchVision Resize transform must have a size specified.")
        if len(tv_resize.size) != 2:
            raise ValueError(
                "TorchVision Resize transform must have a (height, width) "
                f"pair for the size, got {tv_resize.size}."
            )
        return cls(size=tv_resize.size)


@dataclass
class RandomCrop(DecoderTransform):
    """Crop the decoded frame to a given size at a random location in the frame.

    Complementary TorchVision transform: :class:`~torchvision.transforms.v2.RandomCrop`.
    Padding of all kinds is disabled. The random location within the frame is
    determined during the initialization of the
    :class:~`torchcodec.decoders.VideoDecoder` object that owns this transform.
    As a consequence, each decoded frame in the video will be cropped at the
    same location. Videos with variable resolution may result in undefined
    behavior.

    Args:
        size: (sequence of int): Desired output size. Must be a sequence of
            the form (height, width).
    """

    size: Sequence[int]

    # Note that these values are never read by this object or the decoder. We
    # record them for testing purposes only.
    _top: Optional[int] = None
    _left: Optional[int] = None

    def _make_transform_spec(
        self, input_dims: Tuple[Optional[int], Optional[int]]
    ) -> str:
        if len(self.size) != 2:
            raise ValueError(
                f"RandomCrop's size must be a sequence of length 2, got {self.size}. "
                "This should never happen, please report a bug."
            )

        height, width = input_dims
        if height is None:
            raise ValueError(
                "Video metadata has no height. "
                "RandomCrop can only be used when input frame dimensions are known."
            )
        if width is None:
            raise ValueError(
                "Video metadata has no width. "
                "RandomCrop can only be used when input frame dimensions are known."
            )

        # Note: This logic below must match the logic in
        #       torchvision.transforms.v2.RandomCrop.make_params(). Given
        #       the same seed, they should get the same result. This is an
        #       API guarantee with our users.
        if height < self.size[0] or width < self.size[1]:
            raise ValueError(
                f"Input dimensions {input_dims} are smaller than the crop size {self.size}."
            )

        top = int(torch.randint(0, height - self.size[0] + 1, size=()).item())
        self._top = top

        left = int(torch.randint(0, width - self.size[1] + 1, size=()).item())
        self._left = left

        return f"crop, {self.size[0]}, {self.size[1]}, {left}, {top}"

    def _calculate_output_dims(
        self, input_dims: Tuple[Optional[int], Optional[int]]
    ) -> Tuple[Optional[int], Optional[int]]:
        # TODO: establish this invariant in the constructor during refactor
        assert len(self.size) == 2

        height, width = input_dims
        if height is None:
            raise ValueError(
                "Video metadata has no height. "
                "RandomCrop can only be used when input frame dimensions are known."
            )
        if width is None:
            raise ValueError(
                "Video metadata has no width. "
                "RandomCrop can only be used when input frame dimensions are known."
            )

        return (self.size[0], self.size[1])

    @classmethod
    def _from_torchvision(
        cls,
        tv_random_crop: nn.Module,
    ):
        v2 = import_torchvision_transforms_v2()

        if not isinstance(tv_random_crop, v2.RandomCrop):
            raise ValueError(
                "Transform must be TorchVision's RandomCrop, "
                f"it is instead {type(tv_random_crop).__name__}. "
                "This should never happen, please report a bug."
            )

        if tv_random_crop.padding is not None:
            raise ValueError(
                "TorchVision RandomCrop transform must not specify padding."
            )

        if tv_random_crop.pad_if_needed is True:
            raise ValueError(
                "TorchVision RandomCrop transform must not specify pad_if_needed."
            )

        if tv_random_crop.fill != 0:
            raise ValueError("TorchVision RandomCrop fill must be 0.")

        if tv_random_crop.padding_mode != "constant":
            raise ValueError("TorchVision RandomCrop padding_mode must be constant.")

        if len(tv_random_crop.size) != 2:
            raise ValueError(
                "TorchVision RandcomCrop transform must have a (height, width) "
                f"pair for the size, got {tv_random_crop.size}."
            )

        return cls(size=tv_random_crop.size)
