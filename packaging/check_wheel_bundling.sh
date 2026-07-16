#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Safety check for the wheel's bundled native libraries.
#
# The invariant we enforce:
#   1. libjpeg IS bundled (the CPU JPEG image decoder needs it at runtime), and
#   2. NONE of FFmpeg / torch / CUDA are bundled -- those must stay EXTERNAL
#      (FFmpeg is GPL and user-provided; torch/CUDA come from the torch wheel).
#
# We deliberately use a DENYLIST (forbidden patterns) rather than a strict
# "only libjpeg" allowlist: the per-platform repair tools (and pytorch's
# test-infra macOS build, which delocates before our post-script runs) also
# bundle OS / interpreter runtime libs such as libc++ and libpython. Those are
# benign (provided by the OS / the Python interpreter) and we can't reliably
# stop upstream tooling from bundling them, so we don't fail on them -- we only
# fail on the libraries that actually matter for licensing / correctness.
#
# Runs on every wheel-building CI job (Linux/macOS/Windows) via the post-scripts;
# written in portable bash so it works under git-bash on the Windows runners too.

set -euo pipefail

# Substrings that must NOT appear in any bundled library's basename. Covers
# every SONAME/DLL-version spelling across platforms.
FORBIDDEN=(
    # FFmpeg (GPL, provided by the user's runtime FFmpeg install)
    libav avcodec- avformat- avutil- avfilter- avdevice-
    libsw swscale- swresample- libpostproc postproc-
    # PyTorch (provided by the torch wheel)
    libtorch torch_cpu torch_cuda libc10 c10.dll c10_cuda
    # CUDA / NVIDIA (provided by torch's CUDA wheels)
    libcu libnv libcupti cudart nvrtc cublas cudnn
)

shopt -s nullglob
wheels=(dist/*.whl)
if [[ ${#wheels[@]} -eq 0 ]]; then
    echo "check_wheel_bundling.sh: no wheels found in dist/!"
    exit 1
fi

status=0
for wheel in "${wheels[@]}"; do
    echo "Checking bundled libraries in $(basename "${wheel}")"

    # All shared libraries shipped in the wheel (.so / .so.N / .dylib / .dll /
    # .pyd), by basename.
    libs=$(unzip -l "${wheel}" | awk '{print $4}' \
        | grep -iE '\.(so|dylib|dll|pyd)($|\.)' \
        | xargs -r -n1 basename \
        | sort -u || true)

    forbidden_found=""
    found_jpeg=0
    other=""
    while read -r lib; do
        [[ -z "${lib}" ]] && continue
        case "${lib}" in
            libtorchcodec_*) continue ;;          # our own libraries
        esac
        if [[ "${lib}" == libjpeg* || "${lib}" == jpeg*.dll ]]; then
            found_jpeg=1
            continue
        fi
        matched_forbidden=0
        for bad in "${FORBIDDEN[@]}"; do
            if [[ "${lib}" == *"${bad}"* ]]; then
                forbidden_found="${forbidden_found} ${lib}"
                matched_forbidden=1
                break
            fi
        done
        if [[ "${matched_forbidden}" -eq 0 ]]; then
            # Benign OS / interpreter runtime lib (e.g. libc++, libpython,
            # vcruntime): allowed, but list it so it's visible in the log.
            other="${other} ${lib}"
        fi
    done <<< "${libs}"

    if [[ -n "${forbidden_found}" ]]; then
        echo "ERROR: forbidden libraries bundled in the wheel:${forbidden_found}"
        echo "FFmpeg, torch and CUDA libraries must stay external."
        status=1
    fi
    if [[ "${found_jpeg}" -eq 0 ]]; then
        echo "ERROR: libjpeg is NOT bundled in the wheel; the JPEG image decoder"
        echo "would fail at runtime. Was libjpeg-turbo available at build time?"
        status=1
    fi
    if [[ -n "${other}" ]]; then
        echo "Note: also bundled (allowed OS/runtime libs):${other}"
    fi
    if [[ "${status}" -eq 0 ]]; then
        echo "OK: libjpeg bundled; no FFmpeg/torch/CUDA libraries bundled."
    fi
done

exit "${status}"
