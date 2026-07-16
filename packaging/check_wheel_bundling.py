# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Safety check for the wheel's bundled native libraries.

Invariant we enforce:
  1. libjpeg IS bundled (the CPU JPEG image decoder needs it at runtime), and
  2. NONE of FFmpeg / torch / CUDA are bundled -- those must stay EXTERNAL
     (FFmpeg is GPL and user-provided; torch/CUDA come from the torch wheel).

We deliberately use a DENYLIST (forbidden substrings) rather than a strict
"only libjpeg" allowlist: the per-platform repair tools -- and pytorch's
test-infra macOS build, which delocates BEFORE our post-script runs -- also
bundle OS / interpreter runtime libs such as libc++ and libpython. Those are
benign (provided by the OS / the Python interpreter) and we cannot reliably stop
upstream tooling from bundling them, so we only fail on the libraries that
actually matter for licensing / correctness.

Written in Python so it runs identically on Linux/macOS/Windows CI.
"""

import glob
import sys
import zipfile

# Substrings that must NOT appear in any bundled library's filename. Covers
# every SONAME / DLL-version spelling across platforms.
FORBIDDEN = [
    # FFmpeg (GPL, provided by the user's runtime FFmpeg install)
    "libav",
    "avcodec-",
    "avformat-",
    "avutil-",
    "avfilter-",
    "avdevice-",
    "libsw",
    "swscale-",
    "swresample-",
    "libpostproc",
    "postproc-",
    # PyTorch (provided by the torch wheel)
    "libtorch",
    "torch_cpu",
    "torch_cuda",
    "libc10",
    "c10.dll",
    "c10_cuda",
    # CUDA / NVIDIA (provided by torch's CUDA wheels)
    "libcu",
    "libnv",
    "libcupti",
    "cudart",
    "nvrtc",
    "cublas",
    "cudnn",
]

_LIB_SUFFIXES = (".so", ".dylib", ".dll", ".pyd")


def _is_shared_lib(name: str) -> bool:
    base = name.split("/")[-1]
    # Matches libfoo.so, libfoo.so.8, libfoo.8.dylib, foo.dll, foo.pyd, ...
    return (
        any(s in base for s in (".so", ".dylib"))
        or base.endswith((".dll", ".pyd"))
        or ".so." in base
    )


def check_wheel(wheel_path: str) -> bool:
    """Return True if the wheel passes the bundling checks."""
    print(f"Checking bundled libraries in {wheel_path.split('/')[-1]}")
    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()

    libs = sorted({n.split("/")[-1] for n in names if _is_shared_lib(n)})

    forbidden_found = []
    other = []
    found_jpeg = False
    for lib in libs:
        if lib.startswith("libtorchcodec_"):  # our own libraries
            continue
        if lib.startswith("libjpeg") or (
            lib.startswith("jpeg") and lib.endswith(".dll")
        ):
            found_jpeg = True
            continue
        bad = next((p for p in FORBIDDEN if p in lib), None)
        if bad is not None:
            forbidden_found.append(lib)
        else:
            # Benign OS / interpreter runtime lib (libc++, libpython,
            # vcruntime, ...): allowed, but list it so it's visible.
            other.append(lib)

    ok = True
    if forbidden_found:
        print(
            "ERROR: forbidden libraries bundled in the wheel: "
            + " ".join(forbidden_found)
        )
        print("FFmpeg, torch and CUDA libraries must stay external.")
        ok = False
    if not found_jpeg:
        print(
            "ERROR: libjpeg is NOT bundled in the wheel; the JPEG image decoder "
            "would fail at runtime. Was libjpeg-turbo available at build time?"
        )
        ok = False
    if other:
        print("Note: also bundled (allowed OS/runtime libs): " + " ".join(other))
    if ok:
        print("OK: libjpeg bundled; no FFmpeg/torch/CUDA libraries bundled.")
    return ok


def main() -> int:
    wheels = glob.glob("dist/*.whl")
    if not wheels:
        print("check_wheel_bundling.py: no wheels found in dist/!")
        return 1
    return 0 if all(check_wheel(w) for w in wheels) else 1


if __name__ == "__main__":
    sys.exit(main())
