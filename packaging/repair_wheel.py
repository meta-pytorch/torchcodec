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

We bundle some third-party native libraries like libjpeg(-turbo), libpng, zlib,
libwebp (+libsharpyuv) and libavif, while making sure we EXCLUDE FFmpeg
(user-provided at runtime) and torch/CUDA (provided by the torch wheel).

libavif is our slim decode-only build fetched from S3 (see
fetch_avif_from_s3.cmake): the AV1 decoder (dav1d) and libyuv are statically
embedded and no AV1 *encoder* is present, so libavif.so is the only avif-related
lib in the wheel -- there are no separate libaom/libdav1d/librav1e/libSvtAv1Enc
to bundle. check_bundling() asserts this (encoder-absence + wheel-size).

Note the FFmpeg exclusions below spell out each FFmpeg soname rather than using a
"libav" glob, because libavif also starts with "libav" and we do NOT want to
exclude it.
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
        # FFmpeg libs, spelled out rather than "libav*" so we don't accidentally
        # exclude libavif (which we DO want to bundle).
        "libavcodec*",
        "libavdevice*",
        "libavfilter*",
        "libavformat*",
        "libavutil*",
        "libavresample*",
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
    # delocate --exclude matches a substring of the dependency's basename. We
    # spell out the FFmpeg libs rather than using "libav" so we don't
    # accidentally exclude libavif (which we DO want to bundle).
    excludes = []
    for pattern in (
        "libavcodec",
        "libavdevice",
        "libavfilter",
        "libavformat",
        "libavutil",
        "libavresample",
        "libsw",
        "libpostproc",
        "libtorch",
        "libc10",
        "libomp",
    ):
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
    # NOTE: libavif is deliberately NOT sourced here. Unlike the other image DLLs
    # (which come from conda), libavif comes from our slim decode-only S3 build
    # and is placed into the wheel directly by CMake (see
    # make_torchcodec_image_library() and fetch_avif_from_s3.cmake). Its AV1
    # decoder (dav1d) is statically embedded and it ships no AV1 encoders, so
    # there are no separate dav1d/aom/rav1e/svtav1 DLLs to bundle. check_bundling()
    # still asserts libavif is present (and that no encoder DLLs leaked in).

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
    - a wheel does NOT bundle libjpeg, libpng, libwebp, libwebpdemux or libavif.
    - the wheel bundles an AV1 encoder library: our libavif is decode-only, so
      encoders (aom/rav1e/svtav1) must never ship (all platforms).
    - (Linux only) the bundled libjpeg isn't libjpeg-turbo.
    - (Linux only) libtorchcodec_image.so links FFmpeg.
    - (Linux only) the compressed wheel is larger than MAX_LINUX_WHEEL_BYTES: the
      slim decode-only libavif should keep us well under it; a regression likely
      means a fat libavif (with encoders) crept back in.
    """
    # 5 MB. Only enforced on Linux (see _is_avif rationale re: the slim S3
    # libavif). Bump if a legitimate dependency growth pushes us over.
    MAX_LINUX_WHEEL_BYTES = 5 * 1024 * 1024

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
        return lib.startswith(("libz", "zlib"))

    def _is_webp(lib):
        return lib.startswith(("libwebp", "libsharpyuv")) or (
            lib.startswith(("webp", "sharpyuv")) and lib.endswith(".dll")
        )

    def _is_avif(lib):
        # We bundle a slim decode-only libavif from S3 (dav1d + libyuv statically
        # embedded, no encoders), so libavif itself is the ONLY avif-related
        # shared lib we expect -- no separate libaom/libdav1d/librav1e/libsvtav1.
        # On Windows the DLL may drop the "lib" prefix.
        stem = lib.lower()
        return stem.startswith("libavif") or (
            stem.startswith("avif") and stem.endswith(".dll")
        )

    def _is_avif_encoder(lib):
        # AV1 encoder libraries. Our decode-only libavif must never pull these in;
        # their presence means we linked a fat libavif (e.g. conda-forge's)
        # instead of our slim S3 build. libdav1d is the AV1 *decoder* and is
        # statically embedded in our libavif, so a standalone libdav1d is equally
        # unexpected -- we flag it here too.
        stem = lib.lower()
        return stem.startswith(("libaom", "librav1e", "libsvtav1", "libdav1d")) or (
            stem.startswith(("aom", "rav1e", "svtav1", "dav1d"))
            and stem.endswith(".dll")
        )

    def _is_webp_demux(lib):
        # libwebpdemux is a separate lib from the base libwebp; it provides the
        # WebPAnimDecoder API used to decode animated webp files.
        return lib.startswith("libwebpdemux") or (
            lib.startswith("webpdemux") and lib.endswith(".dll")
        )

    def _is_allowed(lib):
        if (
            lib.startswith("libtorchcodec_")
            or _is_jpeg(lib)
            or _is_png(lib)
            or _is_zlib(lib)
            or _is_webp(lib)
            or _is_avif(lib)
        ):
            return True
        # On macOS, test-infra's delocate bundles these OS/interpreter libs before
        # our post-script runs; benign and out of our control.
        if platform.system() == "Darwin" and lib.startswith(("libc++", "libpython")):
            return True
        return False

    # FFmpeg soname prefixes, spelled out rather than "libav" so we don't
    # accidentally match libavif -- which libtorchcodec_image.so legitimately
    # links (see _is_avif). "libsw" covers libswscale/libswresample. Mirrors the
    # explicit lists used for the auditwheel/delocate excludes above.
    _FFMPEG_SONAME_PREFIXES = (
        "libavcodec",
        "libavdevice",
        "libavfilter",
        "libavformat",
        "libavutil",
        "libavresample",
        "libsw",
        "libpostproc",
    )

    def _assert_linux_image_lib_no_ffmpeg(zf):
        """Enforce that libtorchcodec_image.so NOT link FFmpeg (no FFmpeg
        soname in DT_NEEDED; see _FFMPEG_SONAME_PREFIXES).

        We built libtorchcodec_image.so separately from the FFmpeg-dependent
        core{4,5,6,7,8}.so libraries: the whole point is to avoid symbol
        interposition between the bundled image codec libs
        (libjpeg/libpng/libwebp) and the user's FFmpeg: FFmpeg may come with its
        own libjpeg/libpng too!

        This check ensures that we didn't accidentally link FFmpeg into
        libtorchcodec_image.so, which would defeat the purpose of building it
        separately.
        """
        from elftools.elf.elffile import ELFFile

        members = [
            n for n in zf.namelist() if n.rsplit("/", 1)[-1] == "libtorchcodec_image.so"
        ]
        if not members:
            raise RuntimeError(
                "libtorchcodec_image.so not found in wheel; the image decoders "
                "are expected to live in their own shared library."
            )
        elf = ELFFile(io.BytesIO(zf.read(members[0])))
        dynamic = elf.get_section_by_name(".dynamic")
        needed = [t.needed for t in dynamic.iter_tags("DT_NEEDED")] if dynamic else []
        ffmpeg_needed = [n for n in needed if n.startswith(_FFMPEG_SONAME_PREFIXES)]
        if ffmpeg_needed:
            raise RuntimeError(
                "libtorchcodec_image.so must not link FFmpeg, but its DT_NEEDED "
                "lists: " + " ".join(ffmpeg_needed)
            )

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
            if not any(lib.lower().startswith(("libavif", "avif")) for lib in libs):
                raise RuntimeError(f"{wheel.name} does not bundle libavif.")
            if not any(_is_webp_demux(lib) for lib in libs):
                raise RuntimeError(
                    f"{wheel.name} does not bundle libwebpdemux (needed for "
                    "animated webp decoding)."
                )
            # Our libavif is decode-only; the AV1 encoders must never ship. This
            # is a correctness guard, so we enforce it on every platform.
            if encoders := [lib for lib in libs if _is_avif_encoder(lib)]:
                raise RuntimeError(
                    f"{wheel.name} bundles AV1 codec libraries that must not "
                    "ship with our decode-only libavif (they should be "
                    "statically embedded or absent): " + " ".join(encoders)
                )
            if platform.system() == "Linux":
                _assert_linux_libjpeg_is_turbo(zf)
                _assert_linux_image_lib_no_ffmpeg(zf)
                wheel_bytes = wheel.stat().st_size
                if wheel_bytes > MAX_LINUX_WHEEL_BYTES:
                    raise RuntimeError(
                        f"{wheel.name} is {wheel_bytes / 1024 / 1024:.1f} MB "
                        "compressed, over the "
                        f"{MAX_LINUX_WHEEL_BYTES / 1024 / 1024:.0f} MB limit. This "
                        "likely means a fat libavif (with AV1 encoders) was "
                        "bundled instead of our slim decode-only build."
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
