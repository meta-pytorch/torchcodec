# TorchCodec supports 2 building scenarios in the respect to compatibility with
# FFmpeg:
#
# * Building against single FFmpeg version. In this case FFmpeg libraries can
#   be detected by standard pkg-config approach.
# * Building against multiple FFmpeg versions at once. In this case the goal is
#   to build few shared libraries each compatible with specific FFmpeg version.
#   At runtime TochCodec will check current environment and select compatible
#   build of the shared library.
#
# This file contains helper definitions and functions to expose CMake FFmpeg
# targets for both scenarios described above. File defines:
#
# `TORCHCODEC_SUPPORTED_FFMPEG_VERSIONS`
#   CMake list of all FFmpeg major versions supported by TorchCodec. Note that
#   this is a list of FFmpeg versions known to TorchCodec rather than a list
#   of FFmpeg versions available on the current system.
#
# `add_ffmpeg_target(ffmpeg_major_version prefix)`
#   * ffmpeg_major_version - FFmpeg major version for which CMake target needs
#     to be defined
#   * prefix - Path to the FFmpeg installation folder
#
#   This function checks that required FFmpeg objects (includes and libraries)
#   are actually available and defines the following target:
#   * `torchcodec::ffmpeg{$ffmpeg_major_version}`
#
# `add_ffmpeg_target_with_pkg_config(ret_ffmpeg_major_version_var)`
#   * `ret_ffmpeg_major_version_var` - parent scope variable where function
#     will return major version of ffmpeg which was found
#
#   This function searches for the FFmpeg with pkg-config and defines the
#   following target:
#   * `torchcodec::ffmpeg{$ffmpeg_major_version}`
#   where `$ffmpeg_major_version` as major version of the detected FFmpeg.

# All FFmpeg major versions supported by TorchCodec.
set(TORCHCODEC_SUPPORTED_FFMPEG_VERSIONS "4;5;6;7;8")

# Below we define FFmpeg library names we expect to have for each FFmpeg
# major version on each platform we support.
if (UNIX AND NOT APPLE)
    set(
       f4_library_file_names
       libavutil.so.56
       libavcodec.so.58
       libavformat.so.58
       libavdevice.so.58
       libavfilter.so.7
       libswscale.so.5
       libswresample.so.3
    )
    set(
       f5_library_file_names
       libavutil.so.57
       libavcodec.so.59
       libavformat.so.59
       libavdevice.so.59
       libavfilter.so.8
       libswscale.so.6
       libswresample.so.4
    )
    set(
       f6_library_file_names
       libavutil.so.58
       libavcodec.so.60
       libavformat.so.60
       libavdevice.so.60
       libavfilter.so.9
       libswscale.so.7
       libswresample.so.4
    )
    set(
       f7_library_file_names
       libavutil.so.59
       libavcodec.so.61
       libavformat.so.61
       libavdevice.so.61
       libavfilter.so.10
       libswscale.so.8
       libswresample.so.5
    )
    set(
       f8_library_file_names
       libavutil.so.60
       libavcodec.so.62
       libavformat.so.62
       libavdevice.so.62
       libavfilter.so.11
       libswscale.so.9
       libswresample.so.6
    )
elseif (APPLE)
    set(
       f4_library_file_names
       libavutil.56.dylib
       libavcodec.58.dylib
       libavformat.58.dylib
       libavdevice.58.dylib
       libavfilter.7.dylib
       libswscale.5.dylib
       libswresample.3.dylib
    )
    set(
       f5_library_file_names
       libavutil.57.dylib
       libavcodec.59.dylib
       libavformat.59.dylib
       libavdevice.59.dylib
       libavfilter.8.dylib
       libswscale.6.dylib
       libswresample.4.dylib
    )
    set(
       f6_library_file_names
       libavutil.58.dylib
       libavcodec.60.dylib
       libavformat.60.dylib
       libavdevice.60.dylib
       libavfilter.9.dylib
       libswscale.7.dylib
       libswresample.4.dylib
    )
    set(
       f7_library_file_names
       libavutil.59.dylib
       libavcodec.61.dylib
       libavformat.61.dylib
       libavdevice.61.dylib
       libavfilter.10.dylib
       libswscale.8.dylib
       libswresample.5.dylib
    )
    set(
       f8_library_file_names
       libavutil.60.dylib
       libavcodec.62.dylib
       libavformat.62.dylib
       libavdevice.62.dylib
       libavfilter.11.dylib
       libswscale.9.dylib
       libswresample.6.dylib
    )
elseif (WIN32)
    set(
        f4_library_file_names
        avutil.lib
        avcodec.lib
        avformat.lib
        avdevice.lib
        avfilter.lib
        swscale.lib
        swresample.lib
    )
    set(
        f5_library_file_names
        avutil.lib
        avcodec.lib
        avformat.lib
        avdevice.lib
        avfilter.lib
        swscale.lib
        swresample.lib
    )
    set(
        f6_library_file_names
        avutil.lib
        avcodec.lib
        avformat.lib
        avdevice.lib
        avfilter.lib
        swscale.lib
        swresample.lib
    )
    set(
        f7_library_file_names
        avutil.lib
        avcodec.lib
        avformat.lib
        avdevice.lib
        avfilter.lib
        swscale.lib
        swresample.lib
    )
    set(
        f8_library_file_names
        avutil.lib
        avcodec.lib
        avformat.lib
        avdevice.lib
        avfilter.lib
        swscale.lib
        swresample.lib
    )
else()
    message(
        FATAL_ERROR
        "Unsupported operating system: ${CMAKE_SYSTEM_NAME}"
    )
endif()

function(add_ffmpeg_target ffmpeg_major_version prefix)
    # Check that given ffmpeg major version is something we support and error out if
    # it's not.
    list(FIND TORCHCODEC_SUPPORTED_FFMPEG_VERSIONS "${ffmpeg_major_version}" _index)
    if (_index LESS 0)
        message(FATAL_ERROR "FFmpeg version ${ffmpeg_major_version} is not supported")
    endif()
    if (NOT DEFINED prefix)
        message(FATAL_ERROR "No prefix defined calling add_ffmpeg_target()")
    endif()

    set(target "torchcodec::ffmpeg${ffmpeg_major_version}")
    set(incdir "${prefix}/include")
    if (UNIX OR APPLE)
        set(libdir "${prefix}/lib")
    elseif (WIN32)
        set(libdir "${prefix}/bin")
    else()
        message(FATAL_ERROR "Unsupported operating system: ${CMAKE_SYSTEM_NAME}")
    endif()

    list(
        TRANSFORM f${ffmpeg_major_version}_library_file_names
        PREPEND ${libdir}/
        OUTPUT_VARIABLE lib_paths
    )

    message("Adding ${target} target")
    # Verify that ffmpeg includes and libraries actually exist.
    foreach (path IN LISTS incdir lib_paths)
        if (NOT EXISTS "${path}")
            message(FATAL_ERROR "${path} does not exist")
        endif()
    endforeach()

    # Actually define the target
    add_library(${target} INTERFACE IMPORTED)
    target_include_directories(${target} INTERFACE ${incdir})
    target_link_libraries(${target} INTERFACE ${lib_paths})
endfunction()

function(add_ffmpeg_target_with_pkg_config ret_ffmpeg_major_version_var)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(TORCHCODEC_LIBAV REQUIRED IMPORTED_TARGET
        libavdevice
        libavfilter
        libavformat
        libavcodec
        libavutil
        libswresample
        libswscale
    )

    # Split libavcodec's version string by '.' and convert it to a list
    string(REPLACE "." ";" libavcodec_version_list ${TORCHCODEC_LIBAV_libavcodec_VERSION})
    # Get the first element of the list, which is the major version
    list(GET libavcodec_version_list 0 libavcodec_major_version)

    if (${libavcodec_major_version} STREQUAL "58")
        set(ffmpeg_major_version "4")
    elseif (${libavcodec_major_version} STREQUAL "59")
        set(ffmpeg_major_version "5")
    elseif (${libavcodec_major_version} STREQUAL "60")
        set(ffmpeg_major_version "6")
    elseif (${libavcodec_major_version} STREQUAL "61")
        set(ffmpeg_major_version "7")
    elseif (${libavcodec_major_version} STREQUAL "62")
        set(ffmpeg_major_version "8")
    else()
        message(FATAL_ERROR "Unsupported libavcodec version: ${libavcodec_major_version}")
    endif()

    message("Adding torchcodec::ffmpeg${ffmpeg_major_version} target")
    add_library(torchcodec::ffmpeg${ffmpeg_major_version} ALIAS PkgConfig::TORCHCODEC_LIBAV)
    set(${ret_ffmpeg_major_version_var} ${ffmpeg_major_version} PARENT_SCOPE)
endfunction()
