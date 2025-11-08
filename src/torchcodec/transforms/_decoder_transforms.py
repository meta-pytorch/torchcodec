# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class DecoderTransform(ABC):
    """Base class for all decoder transforms.

    A DecoderTransform is a transform that is applied by the decoder before
    returning the decoded frame. The implementation does not live in TorchCodec
    itself, but in the underyling decoder. Applying DecoderTransforms to frames
    should be both faster and more memory efficient than receiving normally
    decoded frames and applying the same kind of transform.

    Most DecoderTransforms have a complementary transform in TorchVision,
    specificially in torchvision.transforms.v2. For such transforms, we ensure
    that:

      1. Default behaviors are the same.
      2. The parameters for the DecoderTransform are a subset of the
         TorchVision transform.
      3. Parameters with the same name control the same behavior and accept a
         subset of the same types.
      4. The difference between the frames returned by a DecoderTransform and
         the complementary TorchVision transform are small.

    All DecoderTranforms are applied in the output pixel format and colorspace.
    """

    @abstractmethod
    def make_params(self) -> str:
        pass


@dataclass
class Resize(DecoderTransform):
    """Resize the decoded frame to a given size.

    Complementary TorchVision transform: torchvision.transforms.v2.Resize.
    Interpolation is always bilinear. Anti-aliasing is always on.

    Args:
        size: (sequence of int): Desired output size. Must be a sequence of
            the form (height, width).
    """

    size: Sequence[int]

    def make_params(self) -> str:
        assert len(self.size) == 2
        return f"resize, {self.size[0]}, {self.size[1]}"
