# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bundle binary dependencies into the wheel.

TorchCodec depends on non-Python libraries: FFmpeg, libjpeg, libtorch, etc.
"repairing" a wheel means bundling those binary dependencies into the wheel so
that the wheel runs standalone on a system that doesn't have those libraries
installed.

We bundle some third-party native libraries like libjpeg(-turbo), while making
sure we EXCLUDE FFmpeg (user-provided at runtime) and torch/CUDA (provided by
the torch wheel).
"""

import os  # only for os.environ / os.pathsep (env vars have no pathlib equivalent)
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DIST_DIR = Path("dist")
REPAIRED_DIR = Path("dist_repaired")


def run(cmd, **kwargs):
    cmd = [str(c) for c in cmd]
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
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(Path(conda_prefix) / "lib"), env.get("LD_LIBRARY_PATH", "")]
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
        env["DYLD_FALLBACK_LIBRARY_PATH"] = os.pathsep.join(
            [str(Path(conda_prefix) / "lib"), env.get("DYLD_FALLBACK_LIBRARY_PATH", "")]
        )
    # delocate --exclude matches a substring of the dependency's basename.
    excludes = []
    for pat in ("libav", "libsw", "libpostproc", "libtorch", "libc10", "libomp"):
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
    bin_dir = Path(os.environ.get("CONDA_PREFIX", "")) / "Library" / "bin"
    jpeg_dlls = sorted(
        set(bin_dir.glob("jpeg*.dll")) | set(bin_dir.glob("libjpeg*.dll"))
    )
    if not jpeg_dlls:
        raise FileNotFoundError(f"No libjpeg DLL found under {bin_dir}")

    for wheel in wheels:
        unpack_dir = REPAIRED_DIR / "unpack"
        if unpack_dir.is_dir():
            shutil.rmtree(unpack_dir)
        run([sys.executable, "-m", "wheel", "unpack", wheel, "-d", unpack_dir])
        pkg_dirs = list(unpack_dir.glob("*/torchcodec"))
        if not pkg_dirs:
            raise FileNotFoundError("torchcodec/ package dir not found in wheel")
        pkg_dir = pkg_dirs[0]
        for dll in jpeg_dlls:
            print(f"bundling {dll} -> {pkg_dir}", flush=True)
            shutil.copy(dll, pkg_dir)
        run([sys.executable, "-m", "wheel", "pack", pkg_dir.parent, "-d", REPAIRED_DIR])
        shutil.rmtree(unpack_dir)


def check_bundling():
    """Raise if:
    - a wheel bundles a lib that's not in the allowlist.
    - a wheel does NOT bundle libjpeg.
    """

    def _is_shared_lib(name):
        base = name.rsplit("/", 1)[-1]
        return ".so" in base or ".dylib" in base or base.endswith((".dll", ".pyd"))

    def _is_jpeg(lib):
        return lib.startswith("libjpeg") or (
            lib.startswith("jpeg") and lib.endswith(".dll")
        )

    def _is_allowed(lib):
        if lib.startswith("libtorchcodec_") or _is_jpeg(lib):
            return True
        # On macOS, test-infra's delocate bundles these OS/interpreter libs before
        # our post-script runs; benign and out of our control.
        if platform.system() == "Darwin" and lib.startswith(("libc++", "libpython")):
            return True
        return False

    for wheel in DIST_DIR.glob("*.whl"):
        print(f"Checking bundled libraries in {wheel.name}")
        with zipfile.ZipFile(wheel) as zf:
            libs = sorted(
                {n.rsplit("/", 1)[-1] for n in zf.namelist() if _is_shared_lib(n)}
            )
        if unexpected := [lib for lib in libs if not _is_allowed(lib)]:
            raise RuntimeError(
                f"Unexpected libraries bundled in {wheel.name}: {' '.join(unexpected)}"
            )
        if not any(_is_jpeg(lib) for lib in libs):
            raise RuntimeError(
                f"libjpeg is NOT bundled in {wheel.name}; the JPEG image decoder "
                "would fail at runtime. Was libjpeg-turbo available at build time?"
            )
        print("OK: only libjpeg (and allowed libs) bundled.")


def main():
    wheels = list(DIST_DIR.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError("No wheels found in dist/.")

    if REPAIRED_DIR.is_dir():
        shutil.rmtree(REPAIRED_DIR)
    REPAIRED_DIR.mkdir(parents=True)

    system = platform.system()
    if system == "Linux":
        repair_linux(wheels)
    elif system == "Darwin":
        repair_macos(wheels)
    elif system == "Windows":
        repair_windows(wheels)
    else:
        raise RuntimeError(f"Unknown platform {system!r}.")

    # Replace the original wheels with the repaired ones.
    for wheel in wheels:
        wheel.unlink()
    for wheel in REPAIRED_DIR.glob("*.whl"):
        shutil.move(str(wheel), str(DIST_DIR))
    shutil.rmtree(REPAIRED_DIR)

    print("Repaired wheels:")
    for wheel in DIST_DIR.glob("*.whl"):
        print(f"  {wheel}")

    check_bundling()


if __name__ == "__main__":
    sys.exit(main())
