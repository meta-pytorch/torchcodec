# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Print a lot of diagnostic info about the HEIC decoder / libheif setup.

Run this in CI (or locally) to understand why HEIC decoding does or doesn't
work: it reports the platform, where libtorchcodec_heic lives, whether libheif
is discoverable, the shared-library dependency resolution (ldd/otool), and the
result of actually trying to load the library and decode.

This never raises: it's a pure diagnostic, meant to be dropped into a CI step
before/around the HEIC tests so the logs are actionable on the next run.
"""

import ctypes.util
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path


def _hr(title):
    print(f"\n===== {title} =====", flush=True)


def _run(cmd):
    print(f"$ {' '.join(cmd)}", flush=True)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(out.stdout, flush=True)
        if out.stderr.strip():
            print("[stderr]", out.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"(command failed: {e!r})", flush=True)


def main():
    _hr("platform / python")
    print("sys.platform:", sys.platform)
    print("platform.platform():", platform.platform())
    print("python:", sys.version.replace("\n", " "))
    for var in ("CONDA_PREFIX", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"):
        print(f"{var}={os.environ.get(var, '<unset>')}")

    _hr("torch / torchcodec import")
    try:
        import torch

        print("torch:", torch.__version__, "at", Path(torch.__file__).parent)
    except Exception:
        traceback.print_exc()

    try:
        import torchcodec

        print(
            "torchcodec:",
            getattr(torchcodec, "__version__", "?"),
            "at",
            Path(torchcodec.__file__).parent,
        )
    except Exception:
        traceback.print_exc()

    _hr("libtorchcodec_heic shared library")
    heic_path = None
    try:
        from torchcodec._internally_replaced_utils import _get_extension_path

        heic_path = _get_extension_path("libtorchcodec_heic")
        print("resolved path:", heic_path)
        print("exists:", Path(heic_path).exists())
    except Exception:
        traceback.print_exc()

    if heic_path and Path(heic_path).exists():
        if sys.platform.startswith("linux"):
            _run(["ldd", heic_path])
            _run(["readelf", "-d", heic_path])
        elif sys.platform == "darwin":
            _run(["otool", "-L", heic_path])

    _hr("libheif discoverability")
    print("ctypes.util.find_library('heif'):", ctypes.util.find_library("heif"))
    for name in ("libheif.so.1", "libheif.so", "libheif.dylib", "libheif.1.dylib"):
        try:
            ctypes.CDLL(name)
            print(f"ctypes.CDLL({name!r}): OK")
        except OSError as e:
            print(f"ctypes.CDLL({name!r}): FAILED ({e})")

    _hr("load_heic_library()")
    try:
        from torchcodec._internally_replaced_utils import load_heic_library

        load_heic_library.cache_clear()
        load_heic_library()
        print("load_heic_library(): OK")
    except Exception:
        traceback.print_exc()

    _hr("heic_is_available() / decode attempt")
    try:
        # Reach the test helper if available (running from repo root).
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from test.utils import GRADIENT_HEIC, heic_is_available

        heic_is_available.cache_clear()
        print("heic_is_available():", heic_is_available())
        try:
            from torchcodec.decoders._image_decoders import decode_heic

            out = decode_heic(str(GRADIENT_HEIC.path))
            print("decode_heic OK, shape:", tuple(out.shape), "dtype:", out.dtype)
        except Exception:
            traceback.print_exc()
    except Exception:
        print("(could not import test.utils; run from repo root for this section)")
        traceback.print_exc()

    print("\n===== end HEIC diagnostics =====", flush=True)


if __name__ == "__main__":
    main()
