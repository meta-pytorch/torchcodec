#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Safety check for the wheel's bundled native libraries.
#
# torchcodec bundles EXACTLY ONE third-party native library: libjpeg (for the
# CPU JPEG image decoder). Everything else it links -- FFmpeg (GPL, provided by
# the user at runtime) and the torch/CUDA libraries (provided by the torch
# wheel) -- must stay EXTERNAL and must NOT end up in the wheel.
#
# This asserts that the only shared libraries shipped in the wheel are:
#   - our own libtorchcodec_* libraries, and
#   - libjpeg
# and that libjpeg is indeed present. Anything else (libav*, libtorch*, libc10*,
# libcu*, ...) sneaking in is a packaging bug and fails the build loudly.
#
# Runs on every wheel-building CI job (Linux/macOS/Windows) and is written in
# portable bash so it works under git-bash on the Windows runners too.

set -euo pipefail

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

    unexpected=""
    found_jpeg=0
    while read -r lib; do
        [[ -z "${lib}" ]] && continue
        case "${lib}" in
            libtorchcodec_*)
                # Our own libraries (core / custom_ops / pybind_ops).
                ;;
            libjpeg*|jpeg*.dll|libjpeg*.dll)
                # The one third-party lib we're allowed to bundle.
                found_jpeg=1
                ;;
            *)
                unexpected="${unexpected} ${lib}"
                ;;
        esac
    done <<< "${libs}"

    if [[ -n "${unexpected}" ]]; then
        echo "ERROR: unexpected third-party libraries bundled in the wheel:${unexpected}"
        echo "Only libjpeg (and torchcodec's own libs) may be bundled; FFmpeg,"
        echo "torch and CUDA libraries must stay external."
        status=1
    fi

    if [[ "${found_jpeg}" -eq 0 ]]; then
        echo "ERROR: libjpeg is NOT bundled in the wheel. The JPEG image decoder"
        echo "would fail at runtime. Was libjpeg-turbo available at build time?"
        status=1
    fi

    if [[ "${status}" -eq 0 ]]; then
        echo "OK: the wheel bundles only libjpeg (plus torchcodec's own libs)."
    fi
done

exit "${status}"
