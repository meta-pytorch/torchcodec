#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script builds a *decode-only* libavif from source and installs it into
# ${LIBAVIF_ROOT}. It is the libavif counterpart of packaging/build_ffmpeg.sh:
# it is meant to be run on CI (see .github/workflows/build_libavif.yaml), tarred
# up, and uploaded to S3, so that wheel builds can download and link against it
# instead of conda-forge's libavif.
#
# WHY: conda-forge's libavif links *every* AV1 codec backend (libaom, librav1e,
# libSvtAv1Enc, libdav1d). Three of those are encoders we never call, and they
# bloat every wheel by ~20MB. libavif itself is not an AV1 codec -- it's the
# AVIF/ISOBMFF format library, and it delegates the actual AV1 decoding to a
# pluggable codec backend chosen at build time. For decoding we need exactly one
# decode-capable backend: dav1d, the fastest software AV1 decoder. We build it
# statically into libavif.so and disable all encoders and CLI apps, so the wheel
# bundles a single self-contained libavif.so (~1-2MB) with zero encoder code.
#
# dav1d uses runtime CPU dispatch (it compiles every ISA variant -- SSE2..AVX-512
# on x86, NEON on ARM -- and selects the best the *user's* CPU supports at
# runtime). So building on any runner is fine and never locks the binary to the
# builder's CPU. Two things must hold for the SIMD to actually be present:
#   1. nasm must be available on x86 (else dav1d silently builds C-only). We
#      assert this below.
#   2. We must NOT inject -march=native / -mavx2 into global CFLAGS, which would
#      auto-vectorize dav1d's *non-dispatched* C fallback and crash old CPUs.
#      We deliberately don't touch CFLAGS here.

set -eux

: "${LIBAVIF_VERSION:?LIBAVIF_VERSION must be set (e.g. 1.4.2)}"
: "${LIBAVIF_ROOT:?LIBAVIF_ROOT must be set (install prefix)}"

# dav1d needs meson+ninja to build, and libavif needs cmake+ninja. Ninja and
# meson aren't guaranteed on the builder images, so install them.
python -m pip install --upgrade meson ninja

# nasm is required for dav1d's x86 assembly. Without it the meson build silently
# falls back to a scalar C-only dav1d (no SIMD -> very slow decode). Fail loudly
# on x86 so a misconfigured runner can't silently ship a crippled decoder.
arch="$(uname -m)"
if [[ "${arch}" == x86_64 || "${arch}" == amd64 || "${arch}" == i*86 ]]; then
    if ! command -v nasm > /dev/null 2>&1; then
        echo "ERROR: nasm not found. dav1d's x86 SIMD requires nasm; without it" \
             "dav1d builds C-only and decode is dramatically slower." >&2
        exit 1
    fi
    nasm --version
fi

archive="https://github.com/AOMediaCodec/libavif/archive/refs/tags/v${LIBAVIF_VERSION}.tar.gz"

build_dir=$(mktemp -d -t libavif-build.XXXXXXXXXX)
cleanup() {
    rm -rf "${build_dir}"
}
trap 'cleanup $?' EXIT

cd "${build_dir}"
curl -LsS -o libavif.tar.gz "${archive}"
mkdir libavif
tar -xf libavif.tar.gz -C libavif --strip-components 1

# Configure a decode-only libavif:
# - AVIF_CODEC_DAV1D=LOCAL: fetch+build dav1d (pinned by libavif) as a static,
#   PIC library and embed it into libavif.so. No separate libdav1d ships.
# - All encoder backends (aom/rav1e/svt) and the extra decoders (libgav1/avm)
#   OFF: they already default OFF, listed here for clarity/safety.
# - Apps/tests/examples OFF: those (and only those) need libjpeg/zlib/png, so we
#   also turn those aux deps OFF -- the core decode library needs none of them.
# - AVIF_LIBYUV=OFF: use libavif's vendored libyuv subset for color conversion,
#   avoiding an extra bundled dependency.
cmake -G Ninja -S libavif -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${LIBAVIF_ROOT}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DAVIF_CODEC_DAV1D=LOCAL \
    -DAVIF_CODEC_AOM=OFF \
    -DAVIF_CODEC_RAV1E=OFF \
    -DAVIF_CODEC_SVT=OFF \
    -DAVIF_CODEC_LIBGAV1=OFF \
    -DAVIF_CODEC_AVM=OFF \
    -DAVIF_LIBYUV=OFF \
    -DAVIF_LIBSHARPYUV=OFF \
    -DAVIF_JPEG=OFF \
    -DAVIF_ZLIBPNG=OFF \
    -DAVIF_LIBXML2=OFF \
    -DAVIF_BUILD_APPS=OFF \
    -DAVIF_BUILD_TESTS=OFF \
    -DAVIF_BUILD_EXAMPLES=OFF

cmake --build build --parallel
cmake --install build

# Ship the license/copyright notices. Everything compiled into libavif.so is
# permissive BSD, but BSD-2/BSD-3 require binary redistributions to reproduce the
# copyright notice and license text -- and this artifact is redistributed (public
# S3 bucket, and ultimately bundled into wheels). Two files cover all of it:
# - libavif's LICENSE is a combined file that covers both libavif itself (BSD-2)
#   AND the vendored libyuv subset we compile in (BSD-3, `Files: third_party/
#   libyuv/*` stanza).
# - dav1d's COPYING (BSD-2) -- dav1d is cloned+statically embedded at build time
#   (AVIF_CODEC_DAV1D=LOCAL), so its license isn't in the libavif source tree; we
#   grab it from the checkout the build produced.
mkdir -p "${LIBAVIF_ROOT}/licenses"
cp libavif/LICENSE "${LIBAVIF_ROOT}/licenses/LICENSE.libavif"
dav1d_copying=$(find build -iname COPYING -path '*dav1d*' 2>/dev/null | head -n1 || true)
if [[ -z "${dav1d_copying}" || ! -f "${dav1d_copying}" ]]; then
    echo "ERROR: could not find dav1d's COPYING license file under the build" \
         "directory; refusing to ship a binary without dav1d's license." >&2
    exit 1
fi
cp "${dav1d_copying}" "${LIBAVIF_ROOT}/licenses/COPYING.dav1d"

ls -R "${LIBAVIF_ROOT}"

# Sanity check: the installed libavif must NOT pull in any external AV1 codec
# (dav1d is static; there must be no libaom/librav1e/libSvtAv1Enc/libdav1d as a
# separate shared dependency).
if [[ "$(uname)" == Linux ]]; then
    lib=$(ls "${LIBAVIF_ROOT}"/lib*/libavif.so.* 2>/dev/null | head -n1 || true)
    if [[ -z "${lib}" ]]; then
        echo "ERROR: no installed libavif.so.* found under ${LIBAVIF_ROOT}." >&2
        exit 1
    fi
    if ldd "${lib}" | grep -Ei 'libaom|librav1e|libSvtAv1|libdav1d'; then
        echo "ERROR: libavif links an external AV1 codec; expected a" \
             "self-contained decode-only build with static dav1d." >&2
        exit 1
    fi
fi
