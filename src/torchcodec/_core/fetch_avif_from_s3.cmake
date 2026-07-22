# This file fetches a decode-only libavif from the torchcodec S3 bucket and
# exposes it as the `avif` CMake target, so wheel builds can link+bundle a tiny
# self-contained libavif (dav1d + libyuv statically embedded, no encoders)
# instead of conda-forge's libavif (which drags in ~20MB of unused AV1 encoder
# libraries). The tarball is built on CI via the build_libavif.yaml workflow.
#
# This mirrors fetch_and_expose_non_gpl_ffmpeg_libs.cmake, but libavif is a
# single version and a single library, so there is no version loop.
#
# The `avif` target it defines is the same one the libavif CONFIG package would
# expose, so the libavif link block in CMakeLists.txt works unchanged.

# Avoid warning: see https://cmake.org/cmake/help/latest/policy/CMP0135.html
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

set(libavif_version "1.4.2")

# NOTE: bump this date and refresh the sha256 hashes below whenever a new libavif
# tarball is uploaded to S3 (see build_libavif.yaml). Compute each hash with
# `sha256sum <platform>/1.4.2.tar.gz`.
set(
    base_url
    https://pytorch.s3.amazonaws.com/torchcodec/libavif/2026-07-22
)

if (LINUX)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(platform_url ${base_url}/linux_aarch64)
        set(avif_sha256 9a377d3183bfabdc2e5f2e2f3969d8c3d605de56640b695832827aa7df8a8cce)
    else()  # assume x86_64
        set(platform_url ${base_url}/linux_x86_64)
        set(avif_sha256 b814cbb53ede0b2e32f4907116f466cd68a789283ba8be149d919fb33d336300)
    endif()
elseif (APPLE)
    set(platform_url ${base_url}/macos_arm64)
    set(avif_sha256 357e8cd4a4198dae41b58b63bedb176337a25707030d83d6cff93c661d42b01a)
elseif (WIN32)
    set(platform_url ${base_url}/windows_x86_64)
    set(avif_sha256 eb30dad70894e6d8b15a677b1cd2560d900c5333820be944cc3a8584765e24ac)
else()
    message(FATAL_ERROR "Unsupported operating system: ${CMAKE_SYSTEM_NAME}")
endif()

FetchContent_Declare(
    avif_s3
    URL ${platform_url}/${libavif_version}.tar.gz
    URL_HASH
    SHA256=${avif_sha256}
)
FetchContent_MakeAvailable(avif_s3)

# avif_s3_SOURCE_DIR was set by FetchContent_MakeAvailable and contains the usual
# include/ and lib/ (and bin/ on Windows) directories (the tarball's single
# top-level libavif/ dir is flattened away by FetchContent).
set(include_dir "${avif_s3_SOURCE_DIR}/include")

# libavif's SOVERSION is 16 (paired with libavif 1.4.2). We link the fully
# SONAME'd file directly, exactly as add_ffmpeg_target() does for FFmpeg.
if (LINUX)
    set(lib_path "${avif_s3_SOURCE_DIR}/lib/libavif.so.16")
elseif (APPLE)
    set(lib_path "${avif_s3_SOURCE_DIR}/lib/libavif.16.dylib")
elseif (WIN32)
    # Import library produced by the mingw build; the DLL lives in bin/.
    set(lib_path "${avif_s3_SOURCE_DIR}/lib/libavif.dll.a")
endif()

foreach (path IN LISTS include_dir lib_path)
    if (NOT EXISTS "${path}")
        message(FATAL_ERROR "${path} does not exist")
    endif()
endforeach()

# The runtime shared library that must be shipped in the wheel. On Linux/macOS
# this is the same SONAME'd file we link against; on Windows we link the .dll.a
# import lib but must ship the actual DLL from bin/. We expose its path so
# make_torchcodec_image_library() can install it into the wheel: the FetchContent
# download dir is gone by the time auditwheel/delocate run, so they can't locate
# libavif there -- we must place it into the package ourselves.
if (LINUX)
    set(avif_runtime_lib "${avif_s3_SOURCE_DIR}/lib/libavif.so.16")
elseif (APPLE)
    set(avif_runtime_lib "${avif_s3_SOURCE_DIR}/lib/libavif.16.dylib")
elseif (WIN32)
    set(avif_runtime_lib "${avif_s3_SOURCE_DIR}/bin/libavif.dll")
endif()
if (NOT EXISTS "${avif_runtime_lib}")
    message(FATAL_ERROR "${avif_runtime_lib} does not exist")
endif()

# Directory holding the runtime lib. On Linux/macOS we set this as the image
# library's INSTALL_RPATH so auditwheel/delocate can resolve libavif at repair
# time (see make_torchcodec_image_library() in CMakeLists.txt). Not needed on
# Windows, where the loader finds the co-located DLL.
if (LINUX OR APPLE)
    get_filename_component(avif_lib_dir "${avif_runtime_lib}" DIRECTORY)
endif()

message(STATUS "Adding libavif (decode-only, from S3) as the `avif` target")
add_library(avif INTERFACE IMPORTED)
target_include_directories(avif INTERFACE ${include_dir})
target_link_libraries(avif INTERFACE ${lib_path})
