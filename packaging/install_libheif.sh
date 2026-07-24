#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs libheif (the optional, user-supplied runtime dependency for HEIC
# decoding) into the CURRENT conda env, and makes it discoverable to the dynamic
# loader for subsequent CI steps. libheif is NOT bundled in our wheels (it's
# LGPL); this just provides it at *runtime* so the HEIC tests can run.
#
# IMPORTANT constraints (why this isn't in the shared install_test_dependencies):
#   * conda's libheif pulls libavif16 -> aom>=3.7, which is UNSATISFIABLE
#     alongside FFmpeg 4's aom 3.6 pin. So only call this in envs with FFmpeg>=5
#     (we deliberately don't test HEIC on the FFmpeg-4 matrix cells).
#   * A post-hoc `conda install` re-solves the whole env, which is fragile in the
#     CUDA containers (stale cuda-toolkit pins). Those jobs install libheif at
#     env-CREATION time (setup-miniconda default-packages) instead, and don't
#     call this script.
#
# This script never fails the build: if libheif can't be installed/loaded, it
# logs loudly and returns 0, so the HEIC tests simply skip (unless the caller
# sets FAIL_WITHOUT_HEIC).

set -x  # trace, but NOT -e: we want to be non-fatal.

echo "===== install_libheif.sh ====="
echo "uname: $(uname -a)"
echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"

if ! command -v conda >/dev/null 2>&1; then
    echo "WARNING: conda not on PATH; skipping libheif install. HEIC tests will skip."
    exit 0
fi

echo "--- packages that could conflict with libheif, before install ---"
conda list 2>/dev/null | grep -iE "ffmpeg|aom|svt-av1|libavif|cuda-toolkit|libheif|libde265" || echo "(none)"

# Install libheif into the SAME env, unless it's already there (the CUDA jobs
# install it at env-creation to avoid a fragile post-hoc re-solve; in that case
# we must NOT `conda install` again, which would re-trigger the re-solve).
if conda list 2>/dev/null | grep -qiE "^libheif[[:space:]]"; then
    echo "libheif already present in this env; skipping install."
elif ! conda install -y libheif -c conda-forge; then
    echo "WARNING: 'conda install libheif' failed. HEIC tests will skip. See the"
    echo "solver output above (a common cause is an FFmpeg-4 aom/svt-av1 pin)."
    exit 0
fi

echo "--- libheif and friends, after install ---"
conda list 2>/dev/null | grep -iE "libheif|libde265|libavif|aom|svt-av1" || true

# Make libheif discoverable to the loader in subsequent steps. On Linux/macOS we
# append the conda lib dir to the loader search path (append, not prepend, so
# torch's own bundled libs keep precedence). On Windows, load_heic_library()
# already adds $CONDA_PREFIX/Library/bin via os.add_dll_directory, so no PATH
# plumbing is needed here.
prefix="${CONDA_PREFIX:?}"
case "$(uname -s)" in
    Linux*)
        libvar=LD_LIBRARY_PATH
        libdir="${prefix}/lib"
        ;;
    Darwin*)
        libvar=DYLD_LIBRARY_PATH
        libdir="${prefix}/lib"
        ;;
    *)
        libvar=""
        libdir="${prefix}/Library/bin"  # Windows (informational only)
        ;;
esac

if [ -n "${libvar}" ]; then
    current="$(printenv "${libvar}" || true)"
    export "${libvar}=${libdir}:${current}"
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "${libvar}=${libdir}:${current}" >> "${GITHUB_ENV}"
        echo "Appended ${libdir} to ${libvar} (persisted via GITHUB_ENV)."
    fi
fi
ls -la "${libdir}/"*heif* 2>/dev/null || echo "(no libheif files found in ${libdir})"

# Verify the HEIC library actually loads with libheif present, and flip on
# FAIL_WITHOUT_HEIC for subsequent steps so a broken setup is a hard failure
# (rather than a silent skip). If verification fails, we DON'T set it, so the
# tests skip instead of failing -- and the diagnostics below explain why.
echo "--- HEIC diagnostics ---"
TORCHCODEC_HEIC_DEBUG=1 python packaging/print_heic_diagnostics.py || true

if TORCHCODEC_HEIC_DEBUG=1 python -c "from torchcodec._internally_replaced_utils import load_heic_library; load_heic_library(); print('LIBHEIF LOAD OK')"; then
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "FAIL_WITHOUT_HEIC=1" >> "${GITHUB_ENV}"
        echo "libheif verified: set FAIL_WITHOUT_HEIC=1 for subsequent steps."
    fi
else
    echo "WARNING: libheif installed but libtorchcodec_heic failed to load. HEIC"
    echo "tests will SKIP. See the diagnostics above for the resolution failure."
fi

echo "===== install_libheif.sh done ====="
