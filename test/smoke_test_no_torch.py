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
import json
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


def main():
    lib_dir = sys.argv[1] if len(sys.argv) > 1 else None
    so_path = _find_pybind_so(lib_dir)
    print(f"Loading torch-free module: {so_path}")

    spec = importlib.util.spec_from_file_location("core_pybind_ops", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # The whole point: torch must not be required.
    assert "torch" not in sys.modules, "torch was imported — not torch-free!"

    def to_numpy(frame):
        # frame is a self-describing _DLPackFrame (implements the __dlpack__ /
        # __dlpack_device__ protocol), so numpy reads the correct device itself.
        return np.from_dlpack(frame)

    decoder = mod.create_decoder(_VIDEO)
    try:
        mod.scan_all_streams(decoder)

        # Metadata (JSON serialized in the torch-free core).
        meta = json.loads(mod.get_json_metadata(decoder))
        container_meta = json.loads(mod.get_container_json_metadata(decoder))
        print(
            f"metadata: numFrames={meta.get('numFramesFromHeader')}, "
            f"fps={meta.get('averageFpsFromHeader'):.2f}, "
            f"numStreams={container_meta.get('numStreams')}"
        )
        assert meta.get("numFramesFromHeader") == 390, meta
        assert container_meta.get("numStreams") == 6, container_meta

        mod.add_video_stream(decoder, "NCHW")

        # Sequential next-frame decode.
        first = to_numpy(mod.get_next_frame(decoder))
        print(f"get_next_frame: shape={first.shape}, dtype={first.dtype}")
        assert first.shape == (3, 270, 480), first.shape
        assert first.dtype == np.uint8

        # Indexed single-frame decode (returns data + pts/duration floats).
        data_capsule, pts, duration = mod.get_frame_at_index(decoder, 10)
        frame10 = to_numpy(data_capsule)
        print(f"get_frame_at_index(10): shape={frame10.shape}, pts={pts:.3f}")
        assert frame10.shape == (3, 270, 480), frame10.shape
        assert pts > 0 and duration > 0

        # Time-based single-frame decode.
        data_capsule, pts, _ = mod.get_frame_played_at(decoder, 1.0)
        played = to_numpy(data_capsule)
        assert played.shape == (3, 270, 480), played.shape

        # Batched range decode: data is 4-D, pts/duration are 1-D.
        data_capsule, pts_capsule, dur_capsule = mod.get_frames_in_range(
            decoder, 0, 10, 2
        )
        batch = to_numpy(data_capsule)
        pts_arr = to_numpy(pts_capsule)
        print(f"get_frames_in_range(0,10,2): batch={batch.shape}, pts={pts_arr.shape}")
        assert batch.shape == (5, 3, 270, 480), batch.shape
        assert pts_arr.shape == (5,), pts_arr.shape
        assert batch.sum() > 0
    finally:
        mod.destroy_decoder(decoder)

    assert "torch" not in sys.modules, "torch was imported during decode!"
    print("TORCH-FREE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
