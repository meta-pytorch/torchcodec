# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bundle libjpeg into the wheel with the standard per-OS wheel-repair tool.

Currently torchcodec links exactly one third-party native lib that must travel
with the wheel: libjpeg (for the CPU JPEG image decoder). We bundle it with:
  - Linux:   auditwheel
  - macOS:   delocate
  - Windows: delvewheel

...while EXCLUDING FFmpeg (GPL, user-provided at runtime) and torch/CUDA
(provided by the torch wheel) so ONLY libjpeg (and future permissive image libs)
gets bundled. After repairing we run check_wheel_bundling.py to assert that.

This is written in Python (not bash) so it runs identically on all three
platforms -- in particular on Windows, where the build workflow runs the
post-script through vc_env_helper.bat in cmd, which can't run a .sh and can't
reliably run a relatively-pathed .bat, but runs `python <script>` fine.
"""

import glob
import os
import platform
import shutil
import subprocess
import sys

REPAIRED_DIR = "dist_repaired"


def run(cmd, **kwargs):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def repair_linux(wheels):
    # -U: the runner may ship an older auditwheel preinstalled; without -U
    # `pip install` is a no-op and we'd run the stale version.
    run([sys.executable, "-m", "pip", "install", "-U", "auditwheel"])
    run(["auditwheel", "--version"])
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        # auditwheel must be able to *find* libjpeg to graft it.
        env["LD_LIBRARY_PATH"] = (
            os.path.join(conda_prefix, "lib")
            + os.pathsep
            + env.get("LD_LIBRARY_PATH", "")
        )
    # SONAME globs cover every FFmpeg version present in a multi-FFmpeg wheel.
    excludes = []
    for pat in (
        "libav*",
        "libsw*",
        "libpostproc*",
        "libtorch*",
        "libc10*",
        "libcu*",
        "libnv*",
        "libcupti*",
    ):
        excludes += ["--exclude", pat]
    for wheel in wheels:
        # No --plat: let auditwheel auto-detect the most-compatible manylinux tag.
        run(["auditwheel", "repair", *excludes, "-w", REPAIRED_DIR, wheel], env=env)


def repair_macos(wheels):
    # -U is important: macOS runners ship an older delocate preinstalled whose
    # --exclude behaved differently; without -U `pip install` is a no-op.
    run([sys.executable, "-m", "pip", "install", "-U", "delocate"])
    run(["delocate-wheel", "--version"])
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        env["DYLD_FALLBACK_LIBRARY_PATH"] = (
            os.path.join(conda_prefix, "lib")
            + os.pathsep
            + env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        )
    # delocate --exclude matches a substring of the dependency's basename.
    excludes = []
    for pat in (
        "libav",
        "libsw",
        "libpostproc",
        "libtorch",
        "libc10",
        "libomp",
    ):
        excludes += ["--exclude", pat]
    for wheel in wheels:
        run(
            [
                "delocate-wheel",
                "-v",
                "--ignore-missing-dependencies",
                *excludes,
                "-w",
                REPAIRED_DIR,
                wheel,
            ],
            env=env,
        )


def repair_windows(wheels):
    # We do NOT use delvewheel here. delvewheel roots its dependency analysis on
    # .pyd extension modules, but libjpeg is linked into
    # libtorchcodec_custom_ops*.dll -- a plain DLL that torchcodec loads
    # standalone at runtime (via ctypes), so it's not reachable in delvewheel's
    # graph and libjpeg would never get vendored (and delvewheel also chokes on
    # our intra-wheel libtorchcodec_core*.dll deps).
    #
    # Instead we do what torchvision does on Windows: copy the libjpeg DLL next
    # to our libs inside the wheel. At load time Windows resolves a DLL's
    # dependencies from the DLL's own directory, so custom_ops finds libjpeg
    # there. We repack with `wheel` so the RECORD is regenerated.
    run([sys.executable, "-m", "pip", "install", "-U", "wheel"])
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    bin_dir = os.path.join(conda_prefix, "Library", "bin")
    jpeg_dlls = sorted(
        set(glob.glob(os.path.join(bin_dir, "jpeg*.dll")))
        | set(glob.glob(os.path.join(bin_dir, "libjpeg*.dll")))
    )
    if not jpeg_dlls:
        raise FileNotFoundError(f"No libjpeg DLL found under {bin_dir}")

    for wheel in wheels:
        unpack_dir = os.path.join(REPAIRED_DIR, "unpack")
        if os.path.isdir(unpack_dir):
            shutil.rmtree(unpack_dir)
        run([sys.executable, "-m", "wheel", "unpack", wheel, "-d", unpack_dir])
        pkg_dirs = glob.glob(os.path.join(unpack_dir, "*", "torchcodec"))
        if not pkg_dirs:
            raise FileNotFoundError("torchcodec/ package dir not found in wheel")
        pkg_dir = pkg_dirs[0]
        for dll in jpeg_dlls:
            print(f"bundling {dll} -> {pkg_dir}", flush=True)
            shutil.copy(dll, pkg_dir)
        unpacked = os.path.dirname(pkg_dir)  # the <name>-<version> dir
        run([sys.executable, "-m", "wheel", "pack", unpacked, "-d", REPAIRED_DIR])
        shutil.rmtree(unpack_dir)


def main():
    wheels = glob.glob(os.path.join("dist", "*.whl"))
    if not wheels:
        print("repair_wheel.py: no wheels found in dist/, nothing to do.")
        return 0

    if os.path.isdir(REPAIRED_DIR):
        shutil.rmtree(REPAIRED_DIR)
    os.makedirs(REPAIRED_DIR)

    system = platform.system()
    if system == "Linux":
        repair_linux(wheels)
    elif system == "Darwin":
        repair_macos(wheels)
    elif system == "Windows":
        repair_windows(wheels)
    else:
        print(f"repair_wheel.py: unknown platform {system!r}, skipping.")
        return 0

    # Replace the original wheels with the repaired ones.
    for wheel in wheels:
        os.remove(wheel)
    for wheel in glob.glob(os.path.join(REPAIRED_DIR, "*.whl")):
        shutil.move(wheel, "dist")
    shutil.rmtree(REPAIRED_DIR)

    print("Repaired wheels:")
    for wheel in glob.glob(os.path.join("dist", "*.whl")):
        print("  " + wheel)

    # Verify the repair bundled ONLY libjpeg (no FFmpeg/torch/CUDA leaked in).
    run([sys.executable, os.path.join("packaging", "check_wheel_bundling.py")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
