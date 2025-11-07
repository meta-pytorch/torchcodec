# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class DecoderNativeTransform(ABC):
    """TODO: docstring"""

    @abstractmethod
    def make_params(self) -> str:
        pass


@dataclass
class Resize(DecoderNativeTransform):
    """
    TODO. One benefit of having parallel definitions is that it gives us a place
          to put documentation about what behavior we do and do not support. For
          example, we don't yet have fields for `interpolation` and `antialias`
          because we don't allow users to control those yet in decoder-native
          transforms.
    """

    # Also note that this type is more restrictive than what TorchVision
    # accepts, but it accurately reflects current decoder-native transform
    # limitations. We can reflect that not just in our docs, but also type
    # annotations.
    size: Sequence[int]

    def make_params(self) -> str:
        assert len(self.size) == 2
        return f"resize, {self.size[0]}, {self.size[1]}"
