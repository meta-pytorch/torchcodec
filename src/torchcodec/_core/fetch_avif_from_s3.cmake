# This file fetches a decode-only libavif from the torchcodec S3 bucket and
# exposes it as the `avif` CMake target, so wheel builds can link+bundle a tiny
# self-contained libavif (dav1d statically embedded, no encoders) instead of
# conda-forge's libavif (which drags in ~20MB of unused AV1 encoder libraries).
# The tarball is built on CI via the build_libavif.yaml workflow.
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
# tarball is uploaded to S3 (see build_libavif.yaml).
set(
    base_url
    https://pytorch.s3.amazonaws.com/torchcodec/libavif/REPLACE_WITH_UPLOAD_DATE
)

if (LINUX)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(platform_url ${base_url}/linux_aarch64)
        set(avif_sha256 REPLACE_WITH_LINUX_AARCH64_SHA256)
    else()  # assume x86_64
        set(platform_url ${base_url}/linux_x86_64)
        set(avif_sha256 REPLACE_WITH_LINUX_X86_64_SHA256)
    endif()
elseif (APPLE)
    set(platform_url ${base_url}/macos_arm64)
    set(avif_sha256 REPLACE_WITH_MACOS_ARM64_SHA256)
elseif (WIN32)
    set(platform_url ${base_url}/windows_x86_64)
    set(avif_sha256 REPLACE_WITH_WINDOWS_X86_64_SHA256)
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

# avif_s3_SOURCE_DIR was set by FetchContent_MakeAvailable and contains the
# usual include/ and lib/ (and bin/ on Windows) directories.
set(include_dir "${avif_s3_SOURCE_DIR}/include")

# libavif's SOVERSION is 16 (paired with libavif 1.4.2). We link the fully
# SONAME'd file directly, exactly as add_ffmpeg_target() does for FFmpeg.
if (LINUX)
    set(lib_dir "${avif_s3_SOURCE_DIR}/lib")
    set(lib_path "${lib_dir}/libavif.so.16")
elseif (APPLE)
    set(lib_dir "${avif_s3_SOURCE_DIR}/lib")
    set(lib_path "${lib_dir}/libavif.16.dylib")
elseif (WIN32)
    # The import library produced by the mingw build; the DLL lives in bin/.
    set(lib_dir "${avif_s3_SOURCE_DIR}/lib")
    set(lib_path "${lib_dir}/libavif.dll.a")
endif()

foreach (path IN LISTS include_dir lib_path)
    if (NOT EXISTS "${path}")
        message(FATAL_ERROR "${path} does not exist")
    endif()
endforeach()

message(STATUS "Adding libavif (decode-only, from S3) as the `avif` target")
add_library(avif INTERFACE IMPORTED)
target_include_directories(avif INTERFACE ${include_dir})
target_link_libraries(avif INTERFACE ${lib_path})
