#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Bundle the permissively-licensed native libraries that get LINKED into
# torchcodec (currently just libjpeg, for the CPU JPEG image decoder) INTO the
# wheel, so `import torchcodec` and `decode_jpeg` work on machines that don't
# have those libraries installed.
#
# We use the standard per-OS wheel-repair tool:
#   - Linux:   auditwheel
#   - macOS:   delocate
#   - Windows: delvewheel
#
# ... but torchcodec deliberately does NOT bundle FFmpeg (GPL, provided by the
# user at runtime) nor the torch/CUDA libraries (provided by the torch wheel).
# So we EXCLUDE all of those, leaving the repair tool to bundle *only* libjpeg
# (and any future permissively-licensed image libs) into <pkg>.libs, with a
# hashed name and rewritten rpaths.
#
# IMPORTANT (Linux): auditwheel drives `patchelf`, and patchelf < 0.18 CORRUPTS
# torchcodec's large torch-linked custom-ops .so when it rewrites the rpath
# (segfault in the library's static initializer at import). `pip install
# patchelf` pins the broken 0.17.2, so we install patchelf >= 0.18 from
# conda-forge and make sure it's the one on PATH.

set -ex

os="$(uname -s)"

repaired_dir="dist_repaired"
rm -rf "${repaired_dir}"
mkdir -p "${repaired_dir}"

shopt -s nullglob
wheels=(dist/*.whl)
if [[ ${#wheels[@]} -eq 0 ]]; then
    echo "repair_wheel.sh: no wheels found in dist/, nothing to do."
    exit 0
fi

case "${os}" in
    Linux)
        python -m pip install auditwheel
        # auditwheel drives `patchelf` (found on PATH; provided by the manylinux
        # build image). NOTE: on a bleeding-edge local toolchain we saw old
        # patchelf (<=0.18.0) corrupt our large torch-linked custom_ops.so ->
        # SIGSEGV in the library's static initializer at import; patchelf 0.19.0
        # was the first version that rewrote it cleanly. We're NOT pinning it
        # here -- letting the CI image use its own patchelf -- because the
        # manylinux toolchain likely doesn't trigger the bug. If a wheel
        # segfaults at import, install a patchelf >= 0.19 before this step, e.g.:
        #   curl -fsSL https://github.com/NixOS/patchelf/releases/download/0.19.0/patchelf-0.19.0-$(uname -m).tar.gz | tar -xz -C /tmp/pf && export PATH=/tmp/pf/bin:$PATH
        patchelf --version || true
        # auditwheel must be able to *find* libjpeg to graft it.
        if [[ -n "${CONDA_PREFIX:-}" ]]; then
            export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        fi
        # SONAME globs cover every FFmpeg version present in a multi-FFmpeg wheel.
        excludes=(
            --exclude "libav*" --exclude "libsw*" --exclude "libpostproc*"
            --exclude "libtorch*" --exclude "libc10*"
            --exclude "libcu*" --exclude "libnv*" --exclude "libcupti*"
        )
        for wheel in "${wheels[@]}"; do
            echo "Repairing (auditwheel) ${wheel}"
            # No --plat: let auditwheel auto-detect the most-compatible manylinux
            # tag from the wheel's symbols.
            auditwheel repair "${excludes[@]}" -w "${repaired_dir}" "${wheel}"
        done
        ;;

    Darwin)
        python -m pip install delocate
        # delocate resolves via install names; conda's are usually absolute, but
        # add the env lib dir to the fallback path to be safe.
        if [[ -n "${CONDA_PREFIX:-}" ]]; then
            export DYLD_FALLBACK_LIBRARY_PATH="${CONDA_PREFIX}/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"
        fi
        # delocate --exclude matches a substring of the dependency's basename.
        # Besides FFmpeg (libav*/libsw*) and torch (libtorch*/libc10*/libomp),
        # we must also exclude libc++ (a macOS system lib, always present at
        # /usr/lib) and libpython (provided by the interpreter). delocate would
        # otherwise copy both into the wheel; auditwheel handles their Linux
        # equivalents automatically (libstdc++/libc are manylinux-allowlisted and
        # libpython is auto-dropped), but delocate needs to be told explicitly.
        # Note: "libtorch" also substring-matches our own libtorchcodec_* libs,
        # which is harmless -- they're the wheel's own libs, not external deps.
        excludes=(
            --exclude libav --exclude libsw --exclude libpostproc
            --exclude libtorch --exclude libc10 --exclude libomp
            --exclude "libc++" --exclude libpython
        )
        for wheel in "${wheels[@]}"; do
            echo "Repairing (delocate) ${wheel}"
            delocate-wheel -v --ignore-missing-dependencies \
                "${excludes[@]}" -w "${repaired_dir}" "${wheel}"
        done
        ;;

    MINGW*|MSYS*|CYGWIN*)
        python -m pip install delvewheel
        add_path=""
        if [[ -n "${CONDA_PREFIX:-}" ]]; then
            # conda ships DLLs under $CONDA_PREFIX/Library/bin on Windows.
            add_path="--add-path ${CONDA_PREFIX}/Library/bin"
        fi
        # delvewheel --exclude takes a comma-separated list; glob patterns are
        # supported. Exclude the DIRECT torch/FFmpeg/CUDA DLL deps by name.
        exclude_list="torch.dll,torch_cpu.dll,torch_cuda.dll,c10.dll,c10_cuda.dll"
        exclude_list="${exclude_list},avcodec-*.dll,avformat-*.dll,avutil-*.dll"
        exclude_list="${exclude_list},avfilter-*.dll,avdevice-*.dll"
        exclude_list="${exclude_list},swscale-*.dll,swresample-*.dll,postproc-*.dll"
        exclude_list="${exclude_list},cudart64_*.dll,nvrtc*.dll,cublas*.dll"
        for wheel in "${wheels[@]}"; do
            echo "Repairing (delvewheel) ${wheel}"
            delvewheel repair ${add_path} --exclude "${exclude_list}" \
                -w "${repaired_dir}" "${wheel}"
        done
        ;;

    *)
        echo "repair_wheel.sh: unknown OS '${os}', skipping wheel repair."
        exit 0
        ;;
esac

# Replace the original wheels with the repaired ones.
rm -f dist/*.whl
mv "${repaired_dir}"/*.whl dist/
rm -rf "${repaired_dir}"

echo "Repaired wheels:"
ls -l dist/*.whl

# Verify the repair bundled ONLY libjpeg (and our own libs) -- no FFmpeg / torch
# / CUDA libs leaked in because an exclude was wrong.
bash packaging/check_wheel_bundling.sh
