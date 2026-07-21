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

We bundle some third-party native libraries like libjpeg(-turbo), libpng, zlib
and libwebp (+libsharpyuv), while making sure we EXCLUDE FFmpeg (user-provided at
runtime) and torch/CUDA (provided by the torch wheel).
"""

import io
import os
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
    run([sys.executable, "-m", "pip", "install", "--upgrade", "auditwheel"])
    run(["auditwheel", "--version"])
    env = os.environ.copy()
    if conda_prefix := env.get("CONDA_PREFIX"):
        # auditwheel must be able to *find* libjpeg to graft it.
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(Path(conda_prefix) / "lib"), env.get("LD_LIBRARY_PATH", "")]
        )

    excludes = []
    for pattern in (
        "libav*",
        "libsw*",
        "libpostproc*",
        "libtorch*",
        "libc10*",
        "libcu*",
        "libnv*",
        "libcupti*",
    ):
        excludes += ["--exclude", pattern]
    for wheel in wheels:
        run(
            ["auditwheel", "repair", *excludes, "--wheel-dir", REPAIRED_DIR, wheel],
            env=env,
        )


def repair_macos(wheels):
    run([sys.executable, "-m", "pip", "install", "--upgrade", "delocate"])
    run(["delocate-wheel", "--version"])
    env = os.environ.copy()
    if conda_prefix := env.get("CONDA_PREFIX"):
        env["DYLD_FALLBACK_LIBRARY_PATH"] = os.pathsep.join(
            [str(Path(conda_prefix) / "lib"), env.get("DYLD_FALLBACK_LIBRARY_PATH", "")]
        )
    # delocate --exclude matches a substring of the dependency's basename.
    excludes = []
    for pattern in ("libav", "libsw", "libpostproc", "libtorch", "libc10", "libomp"):
        excludes += ["--exclude", pattern]

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
    # We do what torchvision does on Windows: copy the libjpeg/libpng/zlib DLLs
    # next to our libs inside the wheel. At load time Windows resolves a DLL's
    # dependencies from the DLL's own directory, so they are found. We repack with
    # `wheel` so the RECORD is regenerated.
    run([sys.executable, "-m", "pip", "install", "-U", "wheel"])
    bin_dir = Path(os.environ.get("CONDA_PREFIX", "")) / "Library" / "bin"

    jpeg_dlls = set(bin_dir.glob("jpeg*.dll")) | set(bin_dir.glob("libjpeg*.dll"))
    if not jpeg_dlls:
        raise FileNotFoundError(f"No libjpeg DLL found under {bin_dir}")
    png_dlls = set(bin_dir.glob("libpng*.dll")) | set(bin_dir.glob("png*.dll"))
    if not png_dlls:
        raise FileNotFoundError(f"No libpng DLL found under {bin_dir}")
    # libpng depends on zlib; bundle it too so libpng can resolve it at load time.
    zlib_dlls = set(bin_dir.glob("zlib*.dll")) | set(bin_dir.glob("libz*.dll"))
    if not zlib_dlls:
        raise FileNotFoundError(f"No zlib DLL found under {bin_dir}")
    # libwebp depends on libsharpyuv; bundle both.
    webp_dlls = set(bin_dir.glob("libwebp*.dll")) | set(bin_dir.glob("webp*.dll"))
    if not webp_dlls:
        raise FileNotFoundError(f"No libwebp DLL found under {bin_dir}")
    sharpyuv_dlls = set(bin_dir.glob("libsharpyuv*.dll")) | set(
        bin_dir.glob("sharpyuv*.dll")
    )
    if not sharpyuv_dlls:
        raise FileNotFoundError(f"No libsharpyuv DLL found under {bin_dir}")

    dlls = sorted(jpeg_dlls | png_dlls | zlib_dlls | webp_dlls | sharpyuv_dlls)

    for wheel in wheels:
        unpack_dir = REPAIRED_DIR / "unpack"
        if unpack_dir.is_dir():
            shutil.rmtree(unpack_dir)
        run([sys.executable, "-m", "wheel", "unpack", wheel, "-d", unpack_dir])
        pkg_dirs = list(unpack_dir.glob("*/torchcodec"))
        if not pkg_dirs:
            raise FileNotFoundError("torchcodec/ package dir not found in wheel")
        pkg_dir = pkg_dirs[0]
        for dll in dlls:
            print(f"bundling {dll} -> {pkg_dir}", flush=True)
            shutil.copy(dll, pkg_dir)
        run([sys.executable, "-m", "wheel", "pack", pkg_dir.parent, "-d", REPAIRED_DIR])
        shutil.rmtree(unpack_dir)


def check_bundling():
    """Raise if:
    - a wheel bundles a lib that's not in the allowlist. This would raise if we
      ever try to bundle FFmpeg or torch/CUDA.
    - a wheel does NOT bundle libjpeg, libpng or libwebp.
    - (Linux only) the bundled libjpeg isn't libjpeg-turbo.
    """

    def _is_shared_lib(name):
        base = name.rsplit("/", 1)[-1]
        return ".so" in base or ".dylib" in base or base.endswith((".dll", ".pyd"))

    def _is_jpeg(lib):
        return lib.startswith("libjpeg") or (
            lib.startswith("jpeg") and lib.endswith(".dll")
        )

    def _is_png(lib):
        return lib.startswith("libpng") or (
            lib.startswith("png") and lib.endswith(".dll")
        )

    def _is_zlib(lib):
        # Linux libz.so.N, macOS libz.N.dylib, Windows zlib*.dll / libz*.dll.
        return lib.startswith(("libz", "zlib"))

    def _is_webp(lib):
        # libwebp itself plus its libsharpyuv dependency (Linux .so, macOS
        # .dylib, Windows libwebp*.dll / webp*.dll).
        return lib.startswith(("libwebp", "libsharpyuv")) or (
            lib.startswith(("webp", "sharpyuv")) and lib.endswith(".dll")
        )

    def _is_allowed(lib):
        if (
            lib.startswith("libtorchcodec_")
            or _is_jpeg(lib)
            or _is_png(lib)
            or _is_zlib(lib)
            or _is_webp(lib)
        ):
            return True
        # On macOS, test-infra's delocate bundles these OS/interpreter libs before
        # our post-script runs; benign and out of our control.
        if platform.system() == "Darwin" and lib.startswith(("libc++", "libpython")):
            return True
        return False

    def _assert_linux_libjpeg_is_turbo(zf):
        jpeg_members = [
            n
            for n in zf.namelist()
            if _is_shared_lib(n) and _is_jpeg(n.rsplit("/", 1)[-1])
        ]
        assert len(jpeg_members) == 1
        jpeg_member = jpeg_members[0]

        from elftools.elf.elffile import ELFFile

        elf = ELFFile(io.BytesIO(zf.read(jpeg_member)))
        verdefs = elf.get_section_by_name(".gnu.version_d")
        is_turbo = verdefs is not None and any(
            aux.name.startswith("LIBJPEGTURBO")
            for _, auxes in verdefs.iter_versions()
            for aux in auxes
        )
        if not is_turbo:
            raise RuntimeError(
                f"Bundled {jpeg_member.rsplit('/', 1)[-1]} is not libjpeg-turbo (no "
                "LIBJPEGTURBO version node). Ensure libjpeg-turbo is the libjpeg "
                "found at build time."
            )

    for wheel in DIST_DIR.glob("*.whl"):
        print(f"Checking bundled libraries in {wheel.name}")
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            libs = sorted({n.rsplit("/", 1)[-1] for n in names if _is_shared_lib(n)})
            if unexpected := [lib for lib in libs if not _is_allowed(lib)]:
                raise RuntimeError(
                    f"Unexpected libraries bundled in {wheel.name}: "
                    + " ".join(unexpected)
                )
            if not any(_is_jpeg(lib) for lib in libs):
                raise RuntimeError(f"{wheel.name} does not bundle libjpeg.")
            if not any(_is_png(lib) for lib in libs):
                raise RuntimeError(f"{wheel.name} does not bundle libpng.")
            if not any(_is_webp(lib) for lib in libs):
                raise RuntimeError(f"{wheel.name} does not bundle libwebp.")
            if platform.system() == "Linux":
                _assert_linux_libjpeg_is_turbo(zf)
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
