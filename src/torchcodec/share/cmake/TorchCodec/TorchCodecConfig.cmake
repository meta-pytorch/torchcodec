# FindTorchCodec
# --------------
#
# Finds the TorchCodec library
#
# This will define the following variables:
#
#   TORCHCODEC_FOUND: True if the system has the TorchCodec library
#   TORCHCODEC_VARIANTS: list of TorchCodec variants. A variant is a supprorted
#   FFmpeg major version.
#
# and the following imported targets:
#
#   torchcodec::ffmpeg${N}
#   torchcodec::core${N}
#
# where N is a TorchCodec variant (FFmpeg major version) from
# TORCHCODEC_VARIANTS list.

include(FindPackageHandleStandardArgs)
include("${CMAKE_CURRENT_LIST_DIR}/ffmpeg_versions.cmake")

# Assume we are in <install-prefix>/share/cmake/TorchCodec/TorchCodecConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(TORCHCODEC_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# Include directories.
set(TORCHCODEC_INCLUDE_DIRS ${TORCHCODEC_INSTALL_PREFIX}/_core)
set(TORCHCODEC_VARIANTS "")

function(add_torchcodec_target ffmpeg_major_version)
    set(target torchcodec::core${ffmpeg_major_version})

    if (NOT TARGET torchcodec::ffmpeg${ffmpeg_major_version})
        message(FATAL_ERROR "torchcodec::ffmpeg${ffmpeg_major_version} target is not defined")
    endif()

    find_library(_LIB_PATH torchcodec_core${ffmpeg_major_version}
        PATHS "${TORCHCODEC_INSTALL_PREFIX}" NO_CACHE NO_DEFAULT_PATH)
    if (NOT _LIB_PATH)
        message(FATAL_ERROR "torchcodec_core${ffmpeg_major_version} shared library is missing")
    endif()

    message("Adding ${target} target")
    add_library(${target} SHARED IMPORTED)
    add_dependencies(${target} torchcodec::ffmpeg${ffmpeg_major_version})
    set_target_properties(${target} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${TORCHCODEC_INCLUDE_DIRS}
      IMPORTED_LOCATION ${_LIB_PATH}
    )
    # Removing _LIB_PATH from cache otherwise it won't be updated
    # on the next call to find_library().
    unset(_LIB_PATH CACHE)

    list(APPEND TORCHCODEC_VARIANTS "${ffmpeg_major_version}")
    set(TORCHCODEC_VARIANTS "${TORCHCODEC_VARIANTS}" PARENT_SCOPE)
endfunction()

set(USE_PKG_CONFIG TRUE)
foreach(ffmpeg_major_version IN LISTS TORCHCODEC_SUPPORTED_FFMPEG_VERSIONS)
    if (DEFINED ENV{TORCHCODEC_FFMPEG${ffmpeg_major_version}_INSTALL_PREFIX})
        add_ffmpeg_target(
            "${ffmpeg_major_version}"
            "$ENV{TORCHCODEC_FFMPEG${ffmpeg_major_version}_INSTALL_PREFIX}"
        )
        add_torchcodec_target(${ffmpeg_major_version})
        set(USE_PKG_CONFIG FALSE)
    endif()
endforeach()

if (USE_PKG_CONFIG)
    # We will get major version of ffmpeg in `ffmpeg_major_version` variable
    add_ffmpeg_target_with_pkg_config(ffmpeg_major_version)
    add_torchcodec_target(${ffmpeg_major_version})
endif()

find_package_handle_standard_args(TorchCodec DEFAULT_MSG TORCHCODEC_VARIANTS)
