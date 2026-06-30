cmake_minimum_required(VERSION 3.18)
project(TorchCodec)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)

# The pybind_ops Python extension module is built with nanobind, in stable-ABI
# (abi3) mode, so we can ship a single wheel that works on Python >= 3.12.
# nanobind requires the `Python` package (not just `Python3`) with the
# Interpreter, Development and Development.SABIModule components; the last one
# provides Python::SABIModule, which nanobind needs for stable-ABI builds.
find_package(Python ${PYTHON_VERSION} EXACT COMPONENTS Interpreter Development Development.SABIModule REQUIRED)
# nanobind ships its CMake config inside the installed Python package; ask it
# where, then load it.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

if(DEFINED TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR AND TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR)
    set(TORCHCODEC_WERROR_OPTION "")
else()
    if (WIN32)
        # TODO set warnings as errors on Windows as well.
        # set(TORCHCODEC_WERROR_OPTION "/WX")
    else()
        set(TORCHCODEC_WERROR_OPTION "-Werror")
    endif()
endif()

if (WIN32)
  # Avoid warnings about non-ASCII characters in source files.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
  # Required for Unicode support in fmt library (bundled with PyTorch).
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
  # Important for when we add Windows CUDA: exporting all symbols is limited to
  # 65535 symbols, which (apparently) will not work for CUDA.
  # https://github.com/pytorch/pytorch/pull/3650
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 ${TORCHCODEC_WERROR_OPTION} ${TORCH_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic ${TORCHCODEC_WERROR_OPTION} ${TORCH_CXX_FLAGS}")
endif()


function(make_torchcodec_sublibrary
    library_name
    type
    sources
    library_dependencies)

    add_library(${library_name} ${type} ${sources})
    set_target_properties(${library_name} PROPERTIES CXX_STANDARD 20)
    target_include_directories(${library_name}
        PRIVATE
        ./../../../
        "${TORCH_INSTALL_PREFIX}/include"
        ${Python3_INCLUDE_DIRS}
    )
    # Use fmt library in header-only mode (bundled with PyTorch): the fmt
    # library is bundled with PyTorch and the headers are available in
    # TORCH_INSTALL_PREFIX/include/, which we add above. To be compatible with
    # torch, we want to rely on the "header-only" mode of these headers and hit
    # all the `#ifdef FMT_HEADER_ONLY` paths, so we define FMT_HEADER_ONLY here.```
    target_compile_definitions(${library_name} PRIVATE FMT_HEADER_ONLY)
    # TC is ABI-stable w.r.t. torch since torch 2.11
    target_compile_definitions(${library_name} PRIVATE TORCH_TARGET_VERSION=0x020b000000000000)

    # Avoid adding the "lib" prefix which we already add explicitly.
    set_target_properties(${library_name} PROPERTIES PREFIX "")

    # On macOS, for wheels, add Homebrew's FFmpeg library path to the rpath so
    # that users with Homebrew-installed FFmpeg can use torchcodec without
    # setting DYLD_LIBRARY_PATH. See
    # https://github.com/pytorch/torchcodec/issues/570.
    # That's the behavior we enable by default when building TorchCodec wheels
    # which we'll ship on PyPI.
    # Note that this means homebrew-FFmpeg will always take precedence over
    # conda-FFmpeg. For this reason, this behavior can (and should!) be disabled
    # when building TorchCodec for conda, by setting
    # TORCHCODEC_DISABLE_HOMEBREW_RPATH.
    # We should have the following behavior for users:
    # - For TorchCodec installed from PyPI (pip or uv): Users don't have to use
    #   a conda env, and by default the homebrew-ffmpeg should be found. For
    #   users who do use a conda env, they should be able to use the
    #   conda-installed FFmpeg, but only if HomeBrew FFmpeg doesn't exist since
    #   homebrew will take precedence.
    # - For TorchCodec installed from conda, users must install FFmpeg from
    #   conda as well (it should be a dependency of the TorchCodec conda
    #   package). Homebrew FFmpeg shouldn't be used.
    if(APPLE AND NOT (DEFINED TORCHCODEC_DISABLE_HOMEBREW_RPATH AND TORCHCODEC_DISABLE_HOMEBREW_RPATH))
        if(DEFINED ENV{HOMEBREW_PREFIX})
            set(HOMEBREW_FFMPEG_LIB "$ENV{HOMEBREW_PREFIX}/opt/ffmpeg/lib")
        else()
            # Default Homebrew location on Apple Silicon
            set(HOMEBREW_FFMPEG_LIB "/opt/homebrew/opt/ffmpeg/lib")
        endif()
        set_target_properties(${library_name} PROPERTIES
            INSTALL_RPATH "${HOMEBREW_FFMPEG_LIB}"
        )
    endif()

    target_link_libraries(
        ${library_name}
        PUBLIC
        ${library_dependencies}
    )

endfunction()

function(make_torchcodec_libraries
    ffmpeg_major_version
    ffmpeg_target)

    # We create three shared libraries per version of FFmpeg, where the version
    # is denoted by N:
    #
    # 1. libtorchcodec_coreN.{ext}: Base library which contains the
    #    implementation of VideoDecoder and everything VideoDecoder needs. On
    #    Linux, {ext} is so. On Mac, it is dylib. On Windows it's dll.
    #
    # 2. libtorchcodec_custom_opsN.{ext}: Implementation of the PyTorch custom
    #    ops. Depends on libtorchcodec_coreN.{ext}. On Linux, {ext} is so.
    #    On Mac, it is dylib. On Windows it's dll.
    #
    # 3. libtorchcodec_pybind_opsN.{ext}: Implementation of the pybind11 ops. We
    #    keep these separate from the PyTorch custom ops because we have to
    #    load these libraries separately on the Python side. Depends on
    #    libtorchcodec_coreN.{ext}. On BOTH Linux and Mac {ext} is so. On
    #    Windows, it's pyd.

    # 1. Create libtorchcodec_coreN.{ext}.
    set(core_library_name "libtorchcodec_core${ffmpeg_major_version}")
    set(core_sources
        AVIOContextHolder.cpp
        AVIOTensorContext.cpp
        AVIOFileContext.cpp
        FFMPEGCommon.cpp
        FilterGraph.cpp
        Frame.cpp
        DeviceInterface.cpp
        CpuDeviceInterface.cpp
        SingleStreamDecoder.cpp
        Encoder.cpp
        ValidationUtils.cpp
        Transform.cpp
        Metadata.cpp
        SwScale.cpp
        WavDecoder.cpp
        NVDECCacheConfig.cpp
        Logging.cpp
    )

    if(ENABLE_CUDA)
        enable_language(CUDA)
	    list(APPEND core_sources CudaDeviceInterface.cpp BetaCudaDeviceInterface.cpp NVDECCache.cpp CUDACommon.cpp NVCUVIDRuntimeLoader.cpp color_conversion.cpp color_conversion.cu)
    endif()

    set(core_library_dependencies
        ${ffmpeg_target}
        ${TORCH_LIBRARIES}
    )

    make_torchcodec_sublibrary(
        "${core_library_name}"
        SHARED
        "${core_sources}"
        "${core_library_dependencies}"
    )

    if(ENABLE_CUDA)
        # We define USE_CUDA to guard CUDA-specific code paths (e.g.
        # NVDECCache usage in NVDECCacheConfig.cpp) and because some torch
        # APIs like aoti_torch_get_current_cuda_stream are only exposed when
        # USE_CUDA is defined.
        # https://github.com/pytorch/pytorch/blob/98e36864e640023a716e058d894ea2d20e76e5f7/torch/csrc/inductor/aoti_torch/c/shim.h#L573-L602
        target_compile_definitions(${core_library_name} PRIVATE USE_CUDA)
        set_property(TARGET ${core_library_name} PROPERTY CUDA_STANDARD 17)
        if(WIN32)
            # CUDA 13.2+ CCCL headers require MSVC's standard-conforming
            # preprocessor when compiling .cu files.
            target_compile_options(${core_library_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
        endif()
    endif()

    # 2. Create libtorchcodec_custom_opsN.{ext}.
    set(custom_ops_library_name "libtorchcodec_custom_ops${ffmpeg_major_version}")
    set(custom_ops_sources
        custom_ops.cpp
    )
    set(custom_ops_dependencies
        ${core_library_name}
        ${Python3_LIBRARIES}
    )
    make_torchcodec_sublibrary(
        "${custom_ops_library_name}"
        SHARED
        "${custom_ops_sources}"
        "${custom_ops_dependencies}"
    )

    # 3. Create libtorchcodec_pybind_opsN — the Python extension module, built
    #    with nanobind in stable-ABI (abi3) mode so a single wheel works on
    #    Python >= 3.12. nanobind_add_module picks the right module type per
    #    platform (e.g. .pyd on Windows), sets the .abi3 suffix, hides symbol
    #    visibility, and links nanobind itself (statically, via NB_STATIC).
    set(pybind_ops_library_name "libtorchcodec_pybind_ops${ffmpeg_major_version}")
    nanobind_add_module(
        ${pybind_ops_library_name}
        STABLE_ABI
        NB_STATIC
        AVIOFileLikeContext.cpp
        pybind_ops.cpp
    )

    # nanobind_add_module doesn't go through make_torchcodec_sublibrary, so apply
    # the same settings those libraries need: include paths, header-only fmt, the
    # torch stable-ABI target version (these files include torch stable headers),
    # and a link against the core library.
    set_target_properties(${pybind_ops_library_name} PROPERTIES CXX_STANDARD 20)
    target_include_directories(${pybind_ops_library_name} PRIVATE
        ./../../../
        "${TORCH_INSTALL_PREFIX}/include"
        ${Python3_INCLUDE_DIRS}
    )
    target_compile_definitions(${pybind_ops_library_name} PRIVATE FMT_HEADER_ONLY)
    target_compile_definitions(${pybind_ops_library_name} PRIVATE TORCH_TARGET_VERSION=0x020b000000000000)
    target_link_libraries(${pybind_ops_library_name} PRIVATE ${core_library_name})

    # Hide symbol visibility in the custom ops library. The nanobind module gets
    # hidden visibility from nanobind_add_module already. Linux only: it's not
    # necessary on Windows, and on Mac hiding visibility can break dynamic casts
    # across shared libraries because the type infos don't get exported.
    if(LINUX)
        target_compile_options(
            ${custom_ops_library_name}
            PUBLIC
            "-fvisibility=hidden"
        )
    endif()

    # We disable the "attributes" warning in the core library.
    # This warning is triggered when, e.g., a class has "default" (i.e.
    # public) visibility but a member of that class has "hidden" visibility.
    # We have such a pattern in the core library with the VideoStreamOptions
    # class and its StableDevice field:
    # - VideoStreamOptions has "default" (public) visibility because it is part
    #   of the core library, and the core library must have public symbols
    #   because it is depended on by other libraries (the pybind ops library and
    #   the custom ops library)
    # - the StableDevice field has "hidden" visibility because this is how
    #   torch::stable exports it.
    # This creates this mismatch where a class has "higher" visibility than its
    # member, hence the warning.
    #
    # We choose to silence this warning here. A possibly better solution would
    # be to have a more fine-grained visibility control where each
    # class/function would be "hidden" unless explicitly marked as public, e.g.
    # through a TORCHCODEC_API annotation. In this case, it is likely that
    # VideoStreamOptions could be in fact "hidden" as well.
    # TODO_STABLE_ABI: do that!
    if (LINUX)
        target_compile_options(
            ${core_library_name}
            PRIVATE
            "-Wno-attributes"
        )
    endif()

    # This is the name passed to NB_MODULE in pybind_ops.cpp. It must match
    # _PYBIND_OPS_MODULE_NAME in torchcodec/_internally_replaced_utils.py, else
    # we won't be able to import the C++ shared library as a Python module at
    # runtime.
    target_compile_definitions(
        ${pybind_ops_library_name}
        PRIVATE
        PYBIND_OPS_MODULE_NAME=core_pybind_ops
    )

    # Install all libraries.
    set(
        all_libraries
        ${core_library_name}
        ${custom_ops_library_name}
        ${pybind_ops_library_name}
    )

    # The install step is invoked within CMakeBuild.build_library() in
    # setup.py and just copies the built files from the temp
    # cmake/setuptools build folder into the CMAKE_INSTALL_PREFIX folder. We
    # still need to manually pass "DESTINATION ..." for cmake to copy those
    # files in CMAKE_INSTALL_PREFIX instead of CMAKE_INSTALL_PREFIX/lib.
    install(
        TARGETS ${all_libraries}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}  # For Windows
    )

endfunction()

if(DEFINED ENV{BUILD_AGAINST_ALL_FFMPEG_FROM_S3})
    message(
        STATUS
        "Building and dynamically linking libtorchcodec against our pre-built
        non-GPL FFmpeg libraries. These libraries are only used at build time,
        you still need a different FFmpeg to be installed for run time!"
    )

    # This will expose the torchcodec::ffmpeg{N} (N=4,5,6,7,8) targets
    include(
        ${CMAKE_CURRENT_SOURCE_DIR}/fetch_and_expose_non_gpl_ffmpeg_libs.cmake
    )

    make_torchcodec_libraries(8 torchcodec::ffmpeg8)
    make_torchcodec_libraries(7 torchcodec::ffmpeg7)
    make_torchcodec_libraries(6 torchcodec::ffmpeg6)
    make_torchcodec_libraries(4 torchcodec::ffmpeg4)
    make_torchcodec_libraries(5 torchcodec::ffmpeg5)
else()
    message(
        STATUS
        "Building and dynamically linking libtorchcodec against the installed
        FFmpeg libraries. This require pkg-config to be installed. If you have
        installed FFmpeg from conda, make sure pkg-config is installed from
        conda as well."
    )

    # This will expose `add_ffmpeg_target_with_pkg_config`
    include("${CMAKE_CURRENT_SOURCE_DIR}/../share/cmake/TorchCodec/ffmpeg_versions.cmake")

    add_ffmpeg_target_with_pkg_config(ffmpeg_major_version)
    make_torchcodec_libraries(${ffmpeg_major_version} torchcodec::ffmpeg${ffmpeg_major_version})

    # Expose these values updwards so that the test compilation does not need
    # to re-figure it out. FIXME: it's not great that we just copy-paste the
    # library names.
    set(libtorchcodec_library_name "libtorchcodec_core${ffmpeg_major_version}" PARENT_SCOPE)
    set(libtorchcodec_custom_ops_name "libtorchcodec_custom_ops${ffmpeg_major_version}" PARENT_SCOPE)
    set(libav_include_dirs ${LIBAV_INCLUDE_DIRS} PARENT_SCOPE)
endif()
