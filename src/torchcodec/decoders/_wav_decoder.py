# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

import torch
from torchcodec import _core
from torchcodec._core._decoder_utils import create_wav_decoder


class WavDecoder:
    # TODO: Docstrings
    def __init__(self, source: str | Path):
        torch._C._log_api_usage_once("torchcodec.decoders.WavDecoder")

        self._decoder = create_wav_decoder(source)
        self._source = source
        self.stream_index = 0

    def get_all_samples(self):
        return _core.get_wav_all_samples(self._decoder)
