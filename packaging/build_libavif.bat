:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: All rights reserved.
::
:: This source code is licensed under the BSD-style license found in the
:: LICENSE file in the root directory of this source tree.

:: Windows counterpart of packaging/build_libavif.sh. Like build_ffmpeg.bat, we
:: build under an msys2/mingw shell so the same POSIX build_libavif.sh drives the
:: decode-only libavif build. libavif is a plain C library, so the resulting
:: avif DLL is loadable/linkable from the MSVC-built torchcodec across the C ABI
:: boundary (same as we do for FFmpeg).
::
:: We install nasm in addition to the toolchain: dav1d's x86 SIMD needs it (see
:: build_libavif.sh). meson+ninja are pip-installed by build_libavif.sh itself.
@echo off

set PROJ_FOLDER=%cd%

choco install -y --no-progress msys2 --package-parameters "/NoUpdate"
C:\tools\msys64\usr\bin\env MSYSTEM=MINGW64 /bin/bash -l -c "pacman -S --noconfirm --needed base-devel mingw-w64-x86_64-toolchain mingw-w64-x86_64-nasm mingw-w64-x86_64-cmake diffutils"
C:\tools\msys64\usr\bin\env MSYSTEM=MINGW64 /bin/bash -l -c "cd ${PROJ_FOLDER} && packaging/build_libavif.sh"

:end
