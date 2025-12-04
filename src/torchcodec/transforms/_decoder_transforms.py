# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from types import ModuleType
from typing import Optional, Sequence, Tuple

import torch
from torch import nn


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
        """Makes the transform spec that is used by the `VideoDecoder`.

        Args:
            input_dims (Tuple[Optional[int], Optional[int]]): The dimensions of
                the input frame in the form (height, width). We cannot know the
                dimensions at object construction time because it's dependent on
                the video being decoded and upstream transforms in the same
                transform pipeline. Not all transforms need to know this; those
                that don't will ignore it. The individual values in the tuple are
                optional because the original values come from file metadata which
                may be missing. We maintain the optionality throughout the APIs so
                that we can decide as late as possible that it's necessary for the
                values to exist. That is, if the values are missing from the
                metadata and we have transforms which ignore the input dimensions,
                we want that to still work.

                Note: This method is the moral equivalent of TorchVision's
                `Transform.make_params()`.

        Returns:
            str: A string which contains the spec for the transform that the
                `VideoDecoder` knows what to do with.
        """
        pass

    def _get_output_dims(self) -> Optional[Tuple[Optional[int], Optional[int]]]:
        """Get the dimensions of the output frame.

        Transforms that change the frame dimensions need to override this
        method. Transforms that don't change the frame dimensions can rely on
        this default implementation.

        Returns:
            Optional[Tuple[Optional[int], Optional[int]]]: The output dimensions.
                - None: The output dimensions are the same as the input dimensions.
                - (int, int): The (height, width) of the output frame.
        """
        return None


def import_torchvision_transforms_v2() -> ModuleType:
    try:
        from torchvision.transforms import v2
    except ImportError as e:
        raise RuntimeError(
            "Cannot import TorchVision; this should never happen, please report a bug."
        ) from e
    return v2


class Resize(DecoderTransform):
    """Resize the decoded frame to a given size.

    Complementary TorchVision transform: :class:`~torchvision.transforms.v2.Resize`.
    Interpolation is always bilinear. Anti-aliasing is always on.

    Args:
        size (Sequence[int]): Desired output size. Must be a sequence of
            the form (height, width).
    """

    def __init__(self, size: Sequence[int]):
        if len(size) != 2:
            raise ValueError(
                "Resize transform must have a (height, width) "
                f"pair for the size, got {size}."
            )
        self.size = size

    def _make_transform_spec(
        self, input_dims: Tuple[Optional[int], Optional[int]]
    ) -> str:
        return f"resize, {self.size[0]}, {self.size[1]}"

    def _get_output_dims(self) -> Optional[Tuple[Optional[int], Optional[int]]]:
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


class CenterCrop(DecoderTransform):
    """Crop the decoded frame to a given size in the center of the frame.

    Complementary TorchVision transform: :class:`~torchvision.transforms.v2.CenterCrop`.

    Args:
        size (Sequence[int]): Desired output size. Must be a sequence of
            the form (height, width).
    """

    def __init__(self, size: Sequence[int]):
        if len(size) != 2:
            raise ValueError(
                "CenterCrop transform must have a (height, width) "
                f"pair for the size, got {size}."
            )
        self.size = size

    def _make_transform_spec(
        self, input_dims: Tuple[Optional[int], Optional[int]]
    ) -> str:
        return f"center_crop, {self.size[0]}, {self.size[1]}"

    def _get_output_dims(self) -> Optional[Tuple[Optional[int], Optional[int]]]:
        return (self.size[0], self.size[1])

    @classmethod
    def _from_torchvision(
        cls,
        tv_center_crop: nn.Module,
    ):
        v2 = import_torchvision_transforms_v2()

        if not isinstance(tv_center_crop, v2.CenterCrop):
            raise ValueError(
                "Transform must be TorchVision's CenterCrop, "
                f"it is instead {type(tv_center_crop).__name__}. "
                "This should never happen, please report a bug."
            )

        if len(tv_center_crop.size) != 2:
            raise ValueError(
                "TorchVision CenterCrop transform must have a (height, width) "
                f"pair for the size, got {tv_center_crop.size}."
            )

        return cls(size=tv_center_crop.size)


class RandomCrop(DecoderTransform):
    """Crop the decoded frame to a given size at a random location in the frame.

    Complementary TorchVision transform: :class:`~torchvision.transforms.v2.RandomCrop`.
    Padding of all kinds is disabled. The random location within the frame is
    determined during the initialization of the
    :class:`~torchcodec.decoders.VideoDecoder` object that owns this transform.
    As a consequence, each decoded frame in the video will be cropped at the
    same location. Videos with variable resolution may result in undefined
    behavior.

    Args:
        size (Sequence[int]): Desired output size. Must be a sequence of
            the form (height, width).
    """

    def __init__(self, size: Sequence[int]):
        if len(size) != 2:
            raise ValueError(
                "RandomCrop transform must have a (height, width) "
                f"pair for the size, got {size}."
            )
        self.size = size

    def _make_transform_spec(
        self, input_dims: Tuple[Optional[int], Optional[int]]
    ) -> str:
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
        left = int(torch.randint(0, width - self.size[1] + 1, size=()).item())

        return f"crop, {self.size[0]}, {self.size[1]}, {left}, {top}"

    def _get_output_dims(self) -> Optional[Tuple[Optional[int], Optional[int]]]:
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
