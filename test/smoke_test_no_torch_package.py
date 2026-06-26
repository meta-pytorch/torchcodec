#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch-free package smoke test.

Requires torchcodec to be installed torch-free (ENABLE_TORCH=0 pip install) in
an environment WITHOUT torch. Verifies the user-facing API works and returns
numpy arrays, with torch never imported.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIDEO = os.path.join(_HERE, "resources", "nasa_13013.mp4")


def main():
    import torchcodec
    from torchcodec.decoders import VideoDecoder

    assert "torch" not in sys.modules, "torch was imported by `import torchcodec`!"

    decoder = VideoDecoder(_VIDEO)

    # Indexing -> numpy.
    frame0 = decoder[0]
    assert isinstance(frame0, np.ndarray), type(frame0)
    assert frame0.shape == (3, 270, 480), frame0.shape
    assert frame0.dtype == np.uint8, frame0.dtype

    # get_frame_at -> Frame with numpy data + float pts.
    frame = decoder.get_frame_at(5)
    assert isinstance(frame.data, np.ndarray), type(frame.data)
    assert frame.data.shape == (3, 270, 480), frame.data.shape
    assert frame.pts_seconds > 0

    # Batched reads.
    assert decoder.get_frames_in_range(0, 10, 2).data.shape == (5, 3, 270, 480)
    assert decoder.get_frames_at([0, 5, 9]).data.shape == (3, 3, 270, 480)
    assert decoder.get_frames_played_at([0.0, 0.5]).data.shape == (2, 3, 270, 480)

    # Metadata.
    assert decoder.metadata.num_frames == 390, decoder.metadata.num_frames
    assert len(decoder) == 390

    assert "torch" not in sys.modules, "torch was imported during decode!"
    print(
        f"torchcodec {torchcodec.__version__ if hasattr(torchcodec, '__version__') else ''} "
        "decoded to numpy without torch."
    )
    print("TORCH-FREE PACKAGE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
