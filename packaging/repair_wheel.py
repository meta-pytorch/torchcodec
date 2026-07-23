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

Because we redistribute those libraries as binaries inside the wheel, their
(permissive) licenses require us to also ship their copyright/license texts.
do that in bundle_third_party_licenses().
"""

import io
import json
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


def _avif_lib_dir():
    # libavif isn't in conda like the other image libs; it's fetched from S3
    # into scikit-build's build dir.
    dirs = [p.resolve() for p in Path("build").glob("*/_deps/avif_s3-src/lib")]
    if len(dirs) != 1:
        raise RuntimeError(f"Expected exactly one S3 libavif dir, found: {dirs}")
    return dirs[0]


def repair_linux(wheels):
    run([sys.executable, "-m", "pip", "install", "--upgrade", "auditwheel"])
    run(["auditwheel", "--version"])
    env = os.environ.copy()
    # for auditwheel to graft libs, it must be able to find them, so we set
    # LD_LIBRARY_PATH: jpeg/png/webp are from conda, libavif is from the S3
    # build dir.
    lib_dirs = [str(_avif_lib_dir())]
    if conda_prefix := env.get("CONDA_PREFIX"):
        lib_dirs.append(str(Path(conda_prefix) / "lib"))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*lib_dirs, env.get("LD_LIBRARY_PATH", "")]
    )

    excludes = []
    for pattern in (
        # FFmpeg libs, spelled out rather than "libav*" so we don't match libavif.
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

    # Same search path as for linux: the libavif install dir for libavif and
    # CONDA_PREFIX for the other image libs.
    search = os.pathsep.join(
        [str(_avif_lib_dir())]
        + ([str(Path(p) / "lib")] if (p := os.environ.get("CONDA_PREFIX")) else [])
    )
    excludes = " ".join(
        f"--exclude {p}"
        for p in (
            "libavcodec",
            "libavdevice",
            "libavfilter",
            "libavformat",
            "libavutil",
            "libavresample",
            "libsw",
            "libpostproc",
            "libtorch.",
            "libtorch_",
            "libc10",
            "libomp",
        )
    )

    for wheel in wheels:
        run(
            [
                "bash",
                "-c",
                # DYLD_LIBRARY_PATH must be set inline on the command ($0=search,
                # $1=wheel): macOS SIP strips it from inherited env on CI (see
                # cibuildwheel #816).
                f'DYLD_LIBRARY_PATH="$0" delocate-wheel -v '
                f'--ignore-missing-dependencies {excludes} -w "{REPAIRED_DIR}" "$1"',
                search,
                str(wheel),
            ]
        )


def repair_windows(wheels):
    # We do what torchvision does on Windows: copy the libjpeg/libpng/zlib etc.
    # DLLs next to our libs inside the wheel. At load time Windows resolves a
    # DLL's dependencies from the DLL's own directory, so they are found. We
    # repack with `wheel` so the RECORD is regenerated.
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
    # libavif comes from our S3 build (not conda): its DLL is in the FetchContent
    # build dir's bin/.
    avif_dlls = set(Path("build").glob("*/_deps/avif_s3-src/bin/libavif*.dll"))
    if not avif_dlls:
        raise FileNotFoundError("No libavif DLL under build/*/_deps/avif_s3-src/bin")

    dlls = sorted(
        jpeg_dlls | png_dlls | zlib_dlls | webp_dlls | sharpyuv_dlls | avif_dlls
    )

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


def bundle_third_party_licenses():
    """Inject the license/copyright texts of the bundled third-party libraries
    into each wheel's .dist-info/licenses/third_party/ dir.

    We redistribute libjpeg-turbo, libpng, zlib, libwebp and libavif (which
    statically embeds dav1d and libyuv) as binaries inside the wheel. Their
    permissive licenses (IJG/BSD/zlib) require reproducing the copyright notice
    and license text in binary redistributions, so we ship them next to our own
    LICENSE.
    """

    def _resolve_conda_licenses():
        """Map dest filename -> source path for the conda-provided image libs.

        conda ships each package's upstream license text under
        <extracted_package_dir>/info/licenses/, and
        CONDA_PREFIX/conda-meta/<pkg>.json records where that dir is. We resolve
        from there so the text always matches the exact binary we bundle.
        """
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            raise RuntimeError(
                "CONDA_PREFIX not set; cannot locate conda license files."
            )
        meta_dir = Path(conda_prefix) / "conda-meta"

        # logical lib -> (dest filename stem, candidate conda package names).
        # Some libs are packaged under more than one name across
        # channels/versions (e.g. zlib vs libzlib), so we try each candidate and
        # take the first that resolves.
        wanted = {
            "libjpeg-turbo": ("LICENSE.libjpeg-turbo", ["libjpeg-turbo"]),
            "libpng": ("LICENSE.libpng", ["libpng"]),
            "zlib": ("LICENSE.zlib", ["libzlib", "zlib"]),
            "libwebp": ("LICENSE.libwebp", ["libwebp", "libwebp-base"]),
        }

        collected = {}
        for logical, (dest_stem, candidates) in wanted.items():
            src_files = None
            for pkg in candidates:
                metas = sorted(meta_dir.glob(f"{pkg}-*.json"))
                if not metas:
                    continue
                info = json.loads(metas[0].read_text())
                lic_dir = Path(info["extracted_package_dir"]) / "info" / "licenses"
                if lic_dir.is_dir():
                    src_files = sorted(f for f in lic_dir.iterdir() if f.is_file())
                    break
            if not src_files:
                raise RuntimeError(
                    f"Could not find license files for {logical} (tried conda "
                    f"packages {candidates} under {meta_dir})."
                )
            # A package usually ships a single license file; if it ships several,
            # keep them all, suffixed with their original name.
            if len(src_files) == 1:
                collected[dest_stem] = src_files[0]
            else:
                for f in src_files:
                    collected[f"{dest_stem}.{f.name}"] = f
        return collected

    def _resolve_avif_licenses():
        """Map dest filename -> source path for the libavif stack (libavif
        itself, plus dav1d and libyuv which are statically embedded inside
        libavif). These are collected into licenses/ by
        packaging/build_libavif.sh and shipped in the S3 artifact that
        fetch_avif_from_s3.cmake unpacks into scikit-build's build dir.
        """
        dirs = [
            p for p in Path("build").glob("*/_deps/avif_s3-src/licenses") if p.is_dir()
        ]
        if not dirs:
            raise RuntimeError(
                "libavif licenses dir not found under "
                "build/*/_deps/avif_s3-src/licenses"
            )
        # Multiple build dirs (one per ABI) may exist; the license texts are
        # identical, so pick any.
        return {f.name: f for f in sorted(dirs[0].iterdir()) if f.is_file()}

    run([sys.executable, "-m", "pip", "install", "-U", "wheel"])
    licenses = {**_resolve_conda_licenses(), **_resolve_avif_licenses()}
    print("Third-party license files to bundle:")
    for name, src in sorted(licenses.items()):
        print(f"  {name} <- {src}")

    scratch = Path("dist_licenses")
    if scratch.is_dir():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    for wheel in sorted(DIST_DIR.glob("*.whl")):
        unpack_dir = scratch / "unpack"
        if unpack_dir.is_dir():
            shutil.rmtree(unpack_dir)
        run([sys.executable, "-m", "wheel", "unpack", wheel, "-d", unpack_dir])
        dist_info_dirs = list(unpack_dir.glob("*/*.dist-info"))
        if len(dist_info_dirs) != 1:
            raise RuntimeError(
                f"Expected exactly one .dist-info in {wheel.name}, "
                f"found: {dist_info_dirs}"
            )
        dest = dist_info_dirs[0] / "licenses" / "third_party"
        dest.mkdir(parents=True, exist_ok=True)
        for name, src in licenses.items():
            shutil.copy(src, dest / name)
        # Repack: `wheel pack` regenerates RECORD so the new files are recorded.
        run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                dist_info_dirs[0].parent,
                "-d",
                scratch,
            ]
        )
        shutil.rmtree(unpack_dir)

    for wheel in DIST_DIR.glob("*.whl"):
        wheel.unlink()
    for wheel in scratch.glob("*.whl"):
        shutil.move(str(wheel), str(DIST_DIR))
    shutil.rmtree(scratch)


def check_bundling():
    """Raise if:
    - a wheel bundles a lib that's not in the allowlist. This would raise if we
      ever try to bundle FFmpeg or torch/CUDA.
    - a wheel does NOT bundle libjpeg, libpng, libwebp, libwebpdemux or libavif.
    - a wheel is missing the license/copyright text of any bundled third-party
      lib under .dist-info/licenses/third_party/ (see
      bundle_third_party_licenses).
    - the wheel bundles an AV1 encoder library: our libavif is decode-only, so
      encoders (aom/rav1e/svtav1) must never ship (all platforms). This is not
      for licensing concern, this is to keep wheel size low.
    - the compressed wheel is larger than MAX_WHEEL_BYTES: the slim decode-only
      libavif should keep us under it.
    - (Linux only) the bundled libjpeg isn't libjpeg-turbo.
    - (Linux only) libtorchcodec_image.so links FFmpeg.
    """
    # 6 MB, bumped to 8 MB for Windows CUDA wheels. Bump if a legitimate
    # dependency growth pushes us over.
    if platform.system() == "Windows" and os.environ.get("ENABLE_CUDA") == "1":
        MAX_WHEEL_BYTES = 8 * 1024 * 1024
    else:
        MAX_WHEEL_BYTES = 6 * 1024 * 1024

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
        stem = lib.lower()
        return stem.startswith("libavif") or (
            stem.startswith("avif") and stem.endswith(".dll")
        )

    def _is_avif_encoder(lib):
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
        if platform.system() == "Darwin" and lib.startswith(("libc++", "libpython")):
            # I can attest libc++ is there, but I'm not entirely sure about
            # libpython. I used to be there when `delocate` was run from the
            # `test-infra` job, but now that we run it here it doesn't seem to
            # be there anymore. I guess it doesn't hurt.
            return True
        return False

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

    def _assert_third_party_licenses(zf):
        """Every bundled third-party lib must ship its license text under
        .dist-info/licenses/third_party/ (see bundle_third_party_licenses)."""
        license_files = [
            n
            for n in zf.namelist()
            if "/licenses/third_party/" in n and not n.endswith("/")
        ]
        # keyword each bundled lib's license file must be identifiable by.
        for keyword in ("jpeg", "png", "zlib", "webp", "avif", "dav1d", "yuv"):
            if not any(keyword in n.lower() for n in license_files):
                raise RuntimeError(
                    f"No third-party license file matching '{keyword}' found in "
                    f".dist-info/licenses/third_party/. Found: {license_files}"
                )

    for wheel in DIST_DIR.glob("*.whl"):
        print(f"Checking bundled libraries in {wheel.name}")
        with zipfile.ZipFile(wheel) as zf:
            _assert_third_party_licenses(zf)
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
            if encoders := [lib for lib in libs if _is_avif_encoder(lib)]:
                raise RuntimeError(
                    f"{wheel.name} bundles AV1 codec libraries that must not "
                    "ship with our decode-only libavif (they should be "
                    "statically embedded or absent): " + " ".join(encoders)
                )
            wheel_bytes = wheel.stat().st_size
            if wheel_bytes > MAX_WHEEL_BYTES:
                raise RuntimeError(
                    f"{wheel.name} is {wheel_bytes / 1024 / 1024:.1f} MB "
                    "compressed, over the "
                    f"{MAX_WHEEL_BYTES / 1024 / 1024:.0f} MB limit. "
                    "Bump MAX_WHEEL_BYTES if a legitimate dependency growth pushes us over. "
                )
            if platform.system() == "Linux":
                _assert_linux_libjpeg_is_turbo(zf)
                _assert_linux_image_lib_no_ffmpeg(zf)
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

    bundle_third_party_licenses()

    print("Repaired wheels:")
    for wheel in DIST_DIR.glob("*.whl"):
        print(f"  {wheel}")

    check_bundling()


if __name__ == "__main__":
    sys.exit(main())
