#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch-free smoke test.

Proves that torchcodec can decode a video WITHOUT PyTorch: it loads the
torch-free pybind module directly (bypassing the torchcodec package, which still
imports torch on the torch path), decodes a frame, and consumes it as a numpy
array via DLPack.

Usage:
    python test/smoke_test_no_torch.py [LIB_DIR]

LIB_DIR is the directory containing libtorchcodec_pybind_ops*.so. If omitted, it
is taken from the TORCHCODEC_LIB_DIR env var, else searched under the repo's
build dirs and the installed torchcodec package.
"""

import glob
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_VIDEO = os.path.join(_HERE, "resources", "nasa_13013.mp4")


def _find_pybind_so(lib_dir):
    search_dirs = []
    if lib_dir:
        search_dirs.append(lib_dir)
    if env_dir := os.environ.get("TORCHCODEC_LIB_DIR"):
        search_dirs.append(env_dir)
    # Common build / install locations.
    search_dirs += [
        os.path.join(_REPO, "build_notorch"),
        os.path.join(_REPO, "src", "torchcodec"),
        os.path.join(_REPO, "src", "torchcodec", "_core"),
    ]
    for d in search_dirs:
        matches = glob.glob(
            os.path.join(d, "**", "libtorchcodec_pybind_ops*.so"), recursive=True
        )
        if matches:
            return sorted(matches)[-1]  # highest FFmpeg version
    raise FileNotFoundError(
        "Could not find libtorchcodec_pybind_ops*.so. Pass LIB_DIR or set "
        "TORCHCODEC_LIB_DIR. Searched: " + ", ".join(search_dirs)
    )


class _DLPackWrapper:
    """Adapts a raw 'dltensor' PyCapsule to numpy's from_dlpack protocol."""

    def __init__(self, capsule):
        self._capsule = capsule

    def __dlpack__(self, *args, **kwargs):
        return self._capsule

    def __dlpack_device__(self):
        return (1, 0)  # (kDLCPU, device 0)


def main():
    lib_dir = sys.argv[1] if len(sys.argv) > 1 else None
    so_path = _find_pybind_so(lib_dir)
    print(f"Loading torch-free module: {so_path}")

    spec = importlib.util.spec_from_file_location("core_pybind_ops", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # The whole point: torch must not be required.
    assert "torch" not in sys.modules, "torch was imported — not torch-free!"

    decoder = mod.create_decoder(_VIDEO)
    try:
        mod.add_video_stream(decoder, "NHWC")
        frame = np.from_dlpack(_DLPackWrapper(mod.get_next_frame(decoder)))
    finally:
        mod.destroy_decoder(decoder)

    print(f"Decoded frame: shape={frame.shape}, dtype={frame.dtype}")
    assert frame.shape == (270, 480, 3), frame.shape
    assert frame.dtype == np.uint8, frame.dtype
    assert frame.sum() > 0, "frame is all zeros"
    assert "torch" not in sys.modules, "torch was imported during decode!"
    print("TORCH-FREE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
