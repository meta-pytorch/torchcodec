# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import importlib
import importlib.util
import os
import shutil
import sys
import traceback
from pathlib import Path
from types import ModuleType

import torch

# Note that this value must match the value used as PYBIND_OPS_MODULE_NAME when we compile _core/pybind_ops.cpp.
# If the values do not match, we will not be able to import the C++ shared library as a Python module at runtime.
_PYBIND_OPS_MODULE_NAME = "core_pybind_ops"


# Copy pasted from torchvision
# https://github.com/pytorch/vision/blob/947ae1dc71867f28021d5bc0ff3a19c249236e2a/torchvision/_internally_replaced_utils.py#L25
def _get_extension_path(lib_name: str) -> str:
    extension_suffixes = []
    if sys.platform == "linux" or sys.platform.startswith("freebsd"):
        extension_suffixes = importlib.machinery.EXTENSION_SUFFIXES
    elif sys.platform == "darwin":
        extension_suffixes = importlib.machinery.EXTENSION_SUFFIXES + [".dylib"]
    elif sys.platform in ("win32", "cygwin"):
        extension_suffixes = importlib.machinery.EXTENSION_SUFFIXES + [".dll", ".pyd"]
    else:
        raise NotImplementedError(f"{sys.platform = } is not not supported")
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        extension_suffixes,
    )

    extfinder = importlib.machinery.FileFinder(
        str(Path(__file__).parent), loader_details
    )
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError(f"No spec found for {lib_name}")

    if ext_specs.origin is None:
        raise ImportError(f"Existing spec found for {lib_name} does not have an origin")

    return ext_specs.origin


def _load_pybind11_module(module_name: str, library_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        module_name,
        library_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load spec or spec.loader for module {module_name} from path {library_path}"
        )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def load_image_library() -> None:
    """Load the FFmpeg-free image library"""
    image_library_path = _get_extension_path("libtorchcodec_image")
    torch.ops.load_library(image_library_path)


@functools.cache
def load_heic_library() -> None:
    """Lazily load the HEIC library (libtorchcodec_heic).

    Unlike the other image decoders (which live in libtorchcodec_image and are
    loaded eagerly at import), the HEIC decoder links libheif, an LGPL library
    we do NOT bundle in our wheels: it's an optional, user-supplied runtime
    dependency, treated like FFmpeg. So this library is loaded lazily, only on
    the first decode_heic (or decode_image of an HEIC file), NEVER at import
    time -- otherwise `import torchcodec` would hard-require libheif.

    On failure (typically because libheif isn't installed / findable) we raise
    an actionable ImportError. This is @functools.cache'd so we only ever load
    once on success; failures aren't cached, so they re-raise on each call.
    """
    heic_library_path = _get_extension_path("libtorchcodec_heic")

    # On Windows there's no LD_LIBRARY_PATH equivalent, so a non-bundled
    # libheif.dll must be discoverable. If we can find it (typically in an
    # active conda env's Library/bin, or on PATH), add its dir to the DLL
    # search path, mirroring how ops.py exposes the FFmpeg DLLs.
    dll_directory_cm = None
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        candidate_dirs = []
        if conda_prefix := os.environ.get("CONDA_PREFIX"):
            candidate_dirs.append(Path(conda_prefix) / "Library" / "bin")
        if heif_on_path := shutil.which("heif-enc") or shutil.which("libheif"):
            candidate_dirs.append(Path(heif_on_path).parent)
        for candidate in candidate_dirs:
            if candidate.is_dir():
                dll_directory_cm = os.add_dll_directory(str(candidate))
                break

    try:
        if dll_directory_cm is not None:
            with dll_directory_cm:
                torch.ops.load_library(heic_library_path)
        else:
            torch.ops.load_library(heic_library_path)
    except Exception as e:
        raise ImportError(
            "Failed to load the HEIC decoding library. HEIC decoding requires "
            "libheif to be installed and discoverable at runtime; TorchCodec "
            "does not bundle it (it's LGPL). Install it, e.g. with "
            "`conda install -c conda-forge libheif`, `apt install libheif1`, or "
            "`brew install libheif`, and make sure it's on your library search "
            f"path. Original error: {e}"
        ) from e


@functools.cache
def load_core_libraries() -> tuple[int, str, ModuleType]:
    """Load the FFmpeg-dependent shared libraries, memoizing the result.

      This raises if the libraries cannot be loaded, typically because FFmpeg
      couldn't be found. This is called:
      - at import time (import torchcodec) within a try/except, to determine
        whether FFmpeg is available, and then load the corresponding ops if it is.
        If FFmpeg isn't available, we don't propagate the exception and
        `import torchcodec` still works: we still want to support the image
        decoders, which don't need FFmpeg.
      - at runtime in all FFmpeg-dependent public entry-points like VideoDecoder,
        so that if FFmpeg isn't available the user gets a clear error message.

      Since this is @functools.cache'd, the libraries are only ever loaded once on
      success. Failures are *not* cached (functools.cache only caches return
      values), so when FFmpeg is missing the load is retried and re-raises on each
      call. That's fine, it only happens on the error path.



    We successively try to load the shared libraries for each version of FFmpeg
    that we support, starting with the highest version and working our way down.
    Once we can load ALL shared libraries for a version of FFmpeg, we have
    succeeded and we stop.

    Note that we use two different methods for loading shared libraries:

      1. torch.ops.load_library(): For PyTorch custom ops and the C++ only
         libraries the custom ops depend on. Loading libraries through PyTorch
         registers the custom ops with PyTorch's runtime and the ops can be
         accessed through torch.ops after loading.

      2. importlib: For pybind11 modules. We load them dynamically, rather
         than using a plain import statement. A plain import statement only
         works when the module name and file name match exactly. Our shared
         libraries do not meet those conditions.
    """
    exceptions = []
    for ffmpeg_major_version in (8, 7, 6, 5, 4):
        core_library_name = f"libtorchcodec_core{ffmpeg_major_version}"
        custom_ops_library_name = f"libtorchcodec_custom_ops{ffmpeg_major_version}"
        pybind_ops_library_name = f"libtorchcodec_pybind_ops{ffmpeg_major_version}"
        try:
            core_library_path = _get_extension_path(core_library_name)
            torch.ops.load_library(core_library_path)
            torch.ops.load_library(_get_extension_path(custom_ops_library_name))

            pybind_ops_library_path = _get_extension_path(pybind_ops_library_name)
            pybind_ops = _load_pybind11_module(
                _PYBIND_OPS_MODULE_NAME, pybind_ops_library_path
            )
            return (ffmpeg_major_version, core_library_path, pybind_ops)
        except Exception:
            # Capture the full traceback for this exception
            exc_traceback = traceback.format_exc()
            exceptions.append((ffmpeg_major_version, exc_traceback))

    traceback_info = (
        "\n[start of libtorchcodec loading traceback]\n"
        + "\n".join(f"FFmpeg version {v}:\n{tb}" for v, tb in exceptions)
        + "[end of libtorchcodec loading traceback]."
    )
    raise RuntimeError(f"""Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version ({torch.__version__}) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:
        """ f"{traceback_info}")
