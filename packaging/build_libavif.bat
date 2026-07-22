:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: All rights reserved.
::
:: This source code is licensed under the BSD-style license found in the
:: LICENSE file in the root directory of this source tree.

@echo off

set PROJ_FOLDER=%cd%

choco install -y --no-progress msys2 --package-parameters "/NoUpdate"
C:\tools\msys64\usr\bin\env MSYSTEM=MINGW64 /bin/bash -l -c "pacman -S --noconfirm --needed base-devel mingw-w64-x86_64-toolchain mingw-w64-x86_64-nasm mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja mingw-w64-x86_64-meson diffutils"
C:\tools\msys64\usr\bin\env MSYSTEM=MINGW64 /bin/bash -l -c "cd ${PROJ_FOLDER} && packaging/build_libavif.sh"

:end
