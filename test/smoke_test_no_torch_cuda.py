#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch-free CUDA smoke test.

Proves that torchcodec can decode a video on the GPU WITHOUT PyTorch: it imports
the package, builds a VideoDecoder on a CUDA device, and gets frames back as
cupy arrays (zero-copy via CUDA DLPack). torch must never be imported.

Requires: a torch-free CUDA build (ENABLE_CUDA=1 ENABLE_TORCH=0), a CUDA GPU,
and cupy installed as the DLPack consumer. Run from a directory other than the
repo root so the installed package (not ./src) is imported.

Usage:
    python test/smoke_test_no_torch_cuda.py [VIDEO_PATH]
"""

import os
import sys

import cupy as cp

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_VIDEO = os.path.join(_HERE, "resources", "nasa_13013.mp4")


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_VIDEO

    import torchcodec
    from torchcodec.decoders import VideoDecoder

    assert "torch" not in sys.modules, "torch was imported — not torch-free!"
    print(f"torchcodec {torchcodec.__version__} (torch not imported)")

    decoder = VideoDecoder(video, device="cuda")
    assert len(decoder) > 0

    # We validate values on the host via cp.asnumpy (a plain device->host copy).
    # We deliberately avoid cupy reductions like .sum() on the device: those
    # JIT-compile a kernel via NVRTC, which exercises cupy's build toolchain
    # rather than torchcodec's decode (and isn't always present in CI).
    def host_sum(arr):
        return int(cp.asnumpy(arr).sum())

    # Single frame -> cupy uint8 on the GPU.
    frame = decoder[10]
    assert isinstance(frame, cp.ndarray), type(frame)
    assert frame.dtype == cp.uint8, frame.dtype
    assert frame.shape == (3, 270, 480), frame.shape
    assert int(frame.device.id) >= 0
    print(
        f"decoder[10]: {type(frame).__module__}.{type(frame).__name__} "
        f"{frame.shape} {frame.dtype} on {frame.device}, sum={host_sum(frame)}"
    )

    # Time-based single frame.
    played = decoder.get_frame_played_at(1.0).data
    assert isinstance(played, cp.ndarray), type(played)
    assert played.shape == (3, 270, 480), played.shape

    # Batched range -> 4-D cupy data, 1-D (host) pts/duration.
    batch = decoder.get_frames_in_range(0, 10, 2)
    assert isinstance(batch.data, cp.ndarray), type(batch.data)
    assert batch.data.shape == (5, 3, 270, 480), batch.data.shape
    assert host_sum(batch.data) > 0
    print(
        f"get_frames_in_range(0,10,2): {batch.data.shape}, "
        f"pts={tuple(batch.pts_seconds)}"
    )

    # Indexed batch.
    at = decoder.get_frames_at([0, 10, 20])
    assert isinstance(at.data, cp.ndarray), type(at.data)
    assert at.data.shape == (3, 3, 270, 480), at.data.shape

    assert "torch" not in sys.modules, "torch was imported during decode!"
    print("TORCH-FREE CUDA SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
