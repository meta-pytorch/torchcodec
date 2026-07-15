:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: All rights reserved.
::
:: This source code is licensed under the BSD-style license found in the
:: LICENSE file in the root directory of this source tree.

:: Taken from torchaudio
@echo on

set VC_VERSION_LOWER=17
set VC_VERSION_UPPER=18

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -legacy -products * -version [%VC_VERSION_LOWER%^,%VC_VERSION_UPPER%^) -property installationPath`) do (
    if exist "%%i" if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15INSTALLDIR=%%i"
        set "VS15VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
        goto vswhere
    )
)

:vswhere
if "%VSDEVCMD_ARGS%" == "" (
    call "%VS15VCVARSALL%" x64 || exit /b 1
) else (
    call "%VS15VCVARSALL%" x64 %VSDEVCMD_ARGS% || exit /b 1
)

@echo on

if "%CU_VERSION%" == "xpu" call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

set DISTUTILS_USE_SDK=1
set BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1

if "%CU_VERSION:~0,2%" == "cu" set ENABLE_CUDA=1
if defined CUDA_PATH set CUDACXX=%CUDA_PATH%\bin\nvcc.exe
:: For CUDA builds, force the Ninja generator. CMake honors the CUDACXX env var
:: (set above to the toolkit matching this build, e.g. v12.6) under Ninja, but
:: NOT under the default Visual Studio generator, which auto-selects the newest
:: nvcc on PATH (e.g. 13.x). A newer nvcc rejects the older gpu archs that torch
:: requests (e.g. compute_50), failing with "nvcc fatal: Unsupported gpu
:: architecture". This mirrors the previous setup.py-based build. We only do
:: this for CUDA builds so the CPU build keeps using the Visual Studio generator.
if defined CUDA_PATH set CMAKE_GENERATOR=Ninja

set args=%1
shift
:start
if [%1] == [] goto done
set args=%args% %1
shift
goto start

:done
if "%args%" == "" (
    echo Usage: vc_env_helper.bat [command] [args]
    echo e.g. vc_env_helper.bat cl /c test.cpp
)

%args% || exit /b 1
