# Single source of truth for torchcodec's _core C++ source file lists.
#
# These lists are consumed by BOTH build systems so that adding or removing a
# source file only requires editing this one file:
# - CMake (the GitHub/OSS build) reads it from CMakeLists.txt in this directory.
# - Buck (the internal Meta build) loads it from src/BUCK.
#
# This file must stay syntactically valid as BOTH Starlark (so Buck can `load()`
# it) and Python (CMake execs it with the Python interpreter to extract the
# lists). Only plain list literals and list comprehensions are allowed here: no
# functions, no `load()`, no `select()`, no f-strings, no Starlark-only or
# Python-only constructs.
#
# Filenames are listed WITHOUT a directory prefix: they live next to this file
# in src/torchcodec/_core/. CMake runs from this directory and uses them as-is;
# Buck prepends the "torchcodec/_core/" path prefix at load time.
#
# The lists are intentionally fine-grained building blocks so each build system
# can compose exactly the targets it needs without changing what gets compiled.
# The two builds group these differently (e.g. CMake compiles FFMPEGCommon.cpp
# into the core library while Buck builds it as a separate target), so do not
# merge the lists.

# CPU sources for the core decoder library. Does NOT include FFMPEGCommon.cpp
# (see ffmpeg_common_sources): the Buck build compiles that into a separate
# "core_common" target.
decoder_core_sources = [
    "AVIOContextHolder.cpp",
    "AVIOTensorContext.cpp",
    "AVIOFileContext.cpp",
    "FilterGraph.cpp",
    "Frame.cpp",
    "DeviceInterface.cpp",
    "CpuDeviceInterface.cpp",
    "Demuxer.cpp",
    "PacketDecoder.cpp",
    "ColorConverter.cpp",
    "SingleStreamDecoder.cpp",
    "Encoder.cpp",
    "ValidationUtils.cpp",
    "Transform.cpp",
    "Metadata.cpp",
    "SwScale.cpp",
    "WavDecoder.cpp",
    "NVDECCacheConfig.cpp",
    "Logging.cpp",
]

# FFmpeg glue. Part of the core library under CMake; a standalone "core_common"
# target under Buck.
ffmpeg_common_sources = [
    "FFMPEGCommon.cpp",
]

# CUDA sources, added to the core library only for CUDA-enabled builds.
decoder_core_cuda_sources = [
    "CudaDeviceInterface.cpp",
    "BetaCudaDeviceInterface.cpp",
    "NVDECCache.cpp",
    "CUDACommon.cpp",
    "NVCUVIDRuntimeLoader.cpp",
    "color_conversion.cpp",
    "color_conversion.cu",
]

# Shared by both the custom-ops and pybind-ops libraries.
file_like_context_sources = [
    "AVIOFileLikeContext.cpp",
]

# PyTorch custom ops registration.
custom_ops_sources = [
    "custom_ops.cpp",
]

# Image decoder implementations. Under CMake these are compiled into a dedicated
# libtorchcodec_image library (NOT the FFmpeg-linked core library) so that our
# bundled image codec libs (libjpeg/libpng/libwebp) load in a separate
# RTLD_LOCAL symbol group from the user's FFmpeg and cannot collide with it. The
# sources are FFmpeg-free (they use only the version-stable torch ABI + the codec
# headers), which is what lets them live in their own library.
image_sources = [
    "DecodeJpeg.cpp",
    "DecodePng.cpp",
    "DecodeWebp.cpp",
    "DecodeGif.cpp",
]

# PyTorch custom-op registration for the image decoders (decode_jpeg/png/webp/
# gif). Kept in its own translation unit so it can be compiled into the
# libtorchcodec_image library alongside image_sources, separately from the
# FFmpeg custom ops in custom_ops.cpp. Registers into the shared torchcodec_ns
# namespace via STABLE_TORCH_LIBRARY_FRAGMENT.
image_ops_sources = [
    "image_custom_ops.cpp",
]

# Vendored giflib (decode-only subset, MIT licensed). Compiled directly from
# source into the libtorchcodec_image library, so the GIF decoder needs no
# external dependency and is always available. See giflib/README for the license
# and local mods.
giflib_sources = [
    "giflib/dgif_lib.c",
    "giflib/gifalloc.c",
    "giflib/gif_hash.c",
    "giflib/openbsd-reallocarray.c",
]

# pybind11 bindings.
pybind_ops_sources = [
    "pybind_ops.cpp",
]
