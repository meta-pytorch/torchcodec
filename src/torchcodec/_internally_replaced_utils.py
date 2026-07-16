# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import importlib.util
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


def load_torchcodec_shared_libraries() -> tuple[int, str, ModuleType]:
    """
    Successively try to load the shared libraries for each version of FFmpeg
    that we support. We always start with the highest version, working our way
    down to the lowest version. Once we can load ALL shared libraries for a
    version of FFmpeg, we have succeeded and we stop.

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
            return ffmpeg_major_version, core_library_path, pybind_ops
        except Exception:
            # Capture the full traceback for this exception
            exc_traceback = traceback.format_exc()
            exceptions.append((ffmpeg_major_version, exc_traceback))

    traceback_info = (
        "\n[start of libtorchcodec loading traceback]\n"
        + "\n".join(f"FFmpeg version {v}:\n{tb}" for v, tb in exceptions)
        + "[end of libtorchcodec loading traceback]."
    )
    raise RuntimeError(
        f"""Could not load libtorchcodec. Likely causes:
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
        """
        f"{traceback_info}"
    )
