# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

import torch
from torchcodec import _core


def _try_create_wav_decoder(source):
    try:
        if isinstance(source, str):
            return _core.create_wav_decoder_from_file(source)
        elif isinstance(source, Path):
            return _core.create_wav_decoder_from_file(str(source))
        else:
            return None
    except Exception as e:
        print(f"Error occurred while processing WAV file: {e}")
        return None


class WavDecoder:
    # TODO: Docstrings
    def __init__(self, source: str | Path):
        torch._C._log_api_usage_once("torchcodec.decoders.WavDecoder")

        self._decoder = _try_create_wav_decoder(source)
        if self._decoder is None:
            raise ValueError(
                "Source is not a supported uncompressed WAV file. "
                "For compressed audio formats or non-WAV files, use AudioDecoder instead."
            )

        self._source = source
        self.stream_index = 0

    def get_all_samples(self):
        return _core.get_wav_all_samples(self._decoder)
