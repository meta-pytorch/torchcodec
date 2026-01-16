# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import io
import json
import os
import shutil
import sys
import traceback
import warnings
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType

import torch
from torch.library import get_ctx, register_fake

from torchcodec._internally_replaced_utils import (  # @manual=//pytorch/torchcodec/src:internally_replaced_utils
    _get_extension_path,
    _get_pybind_ops_module_name,
    _load_pybind11_module,
)

_pybind_ops: ModuleType | None = None


def load_torchcodec_shared_libraries() -> tuple[int, str]:
    # Successively try to load the shared libraries for each version of FFmpeg
    # that we support. We always start with the highest version, working our way
    # down to the lowest version. Once we can load ALL shared libraries for a
    # version of FFmpeg, we have succeeded and we stop.
    #
    # Note that we use two different methods for loading shared libraries:
    #
    #   1. torch.ops.load_library(): For PyTorch custom ops and the C++ only
    #      libraries the custom ops depend on. Loading libraries through PyTorch
    #      registers the custom ops with PyTorch's runtime and the ops can be
    #      accessed through torch.ops after loading.
    #
    #   2. importlib: For pybind11 modules. We load them dynamically, rather
    #      than using a plain import statement. A plain import statement only
    #      works when the module name and file name match exactly. Our shared
    #      libraries do not meet those conditions.

    exceptions = []
    for ffmpeg_major_version in (8, 7, 6, 5, 4):
        pybind_ops_module_name = _get_pybind_ops_module_name(ffmpeg_major_version)
        core_library_name = f"libtorchcodec_core{ffmpeg_major_version}"
        custom_ops_library_name = f"libtorchcodec_custom_ops{ffmpeg_major_version}"
        pybind_ops_library_name = f"libtorchcodec_pybind_ops{ffmpeg_major_version}"
        try:
            core_library_path = _get_extension_path(core_library_name)
            torch.ops.load_library(core_library_path)
            torch.ops.load_library(_get_extension_path(custom_ops_library_name))

            pybind_ops_library_path = _get_extension_path(pybind_ops_library_name)
            global _pybind_ops
            _pybind_ops = _load_pybind11_module(
                pybind_ops_module_name, pybind_ops_library_path
            )
            return ffmpeg_major_version, core_library_path
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


expose_ffmpeg_dlls = nullcontext
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # On windows we try to locate the FFmpeg DLLs and temporarily add them to
    # the DLL search path. This seems to be needed on some users machine, but
    # not on our CI. We don't know why.
    if ffmpeg_path := shutil.which("ffmpeg"):

        def expose_ffmpeg_dlls():  # noqa: F811
            ffmpeg_dir = Path(ffmpeg_path).parent
            return os.add_dll_directory(str(ffmpeg_dir))  # that's the actual CM


with expose_ffmpeg_dlls():
    ffmpeg_major_version, core_library_path = load_torchcodec_shared_libraries()


# Note: We use disallow_in_graph because PyTorch does constant propagation of
# factory functions.
create_from_file = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.create_from_file.default
)
encode_audio_to_file = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.encode_audio_to_file.default
)
encode_audio_to_tensor = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.encode_audio_to_tensor.default
)
_encode_audio_to_file_like = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns._encode_audio_to_file_like.default
)
encode_video_to_file = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.encode_video_to_file.default
)
encode_video_to_tensor = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.encode_video_to_tensor.default
)
_encode_video_to_file_like = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns._encode_video_to_file_like.default
)
create_from_tensor = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.create_from_tensor.default
)
_create_from_file_like = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns._create_from_file_like.default
)
add_video_stream = torch.ops.torchcodec_ns.add_video_stream.default
_add_video_stream = torch.ops.torchcodec_ns._add_video_stream.default
add_audio_stream = torch.ops.torchcodec_ns.add_audio_stream.default
seek_to_pts = torch.ops.torchcodec_ns.seek_to_pts.default
get_next_frame = torch.ops.torchcodec_ns.get_next_frame.default
get_frame_at_pts = torch.ops.torchcodec_ns.get_frame_at_pts.default
get_frame_at_index = torch.ops.torchcodec_ns.get_frame_at_index.default
_get_frames_at_indices_tensor_input = (
    torch.ops.torchcodec_ns.get_frames_at_indices.default
)
_get_frames_by_pts_tensor_input = torch.ops.torchcodec_ns.get_frames_by_pts.default
get_frames_in_range = torch.ops.torchcodec_ns.get_frames_in_range.default
get_frames_by_pts_in_range = torch.ops.torchcodec_ns.get_frames_by_pts_in_range.default
get_frames_by_pts_in_range_audio = (
    torch.ops.torchcodec_ns.get_frames_by_pts_in_range_audio.default
)
get_json_metadata = torch.ops.torchcodec_ns.get_json_metadata.default
_test_frame_pts_equality = torch.ops.torchcodec_ns._test_frame_pts_equality.default
_get_container_json_metadata = (
    torch.ops.torchcodec_ns.get_container_json_metadata.default
)
_get_key_frame_indices = torch.ops.torchcodec_ns._get_key_frame_indices.default
scan_all_streams_to_update_metadata = (
    torch.ops.torchcodec_ns.scan_all_streams_to_update_metadata.default
)
_get_stream_json_metadata = torch.ops.torchcodec_ns.get_stream_json_metadata.default
_get_json_ffmpeg_library_versions = (
    torch.ops.torchcodec_ns._get_json_ffmpeg_library_versions.default
)
_get_backend_details = torch.ops.torchcodec_ns._get_backend_details.default


# =============================
# Functions not related to custom ops, but similar implementation to c++ ops
# =============================
def create_from_bytes(video_bytes: bytes, seek_mode: str | None = None) -> torch.Tensor:
    with warnings.catch_warnings():
        # Ignore warning stating that the underlying video_bytes buffer is
        # non-writable.
        warnings.filterwarnings("ignore", category=UserWarning)
        buffer = torch.frombuffer(video_bytes, dtype=torch.uint8)
    return create_from_tensor(buffer, seek_mode)


def create_from_file_like(
    file_like: io.RawIOBase | io.BufferedReader, seek_mode: str | None = None
) -> torch.Tensor:
    assert _pybind_ops is not None
    return _create_from_file_like(
        _pybind_ops.create_file_like_context(
            file_like, False  # False means not for writing
        ),
        seek_mode,
    )


def encode_audio_to_file_like(
    samples: torch.Tensor,
    sample_rate: int,
    format: str,
    file_like: io.RawIOBase | io.BufferedIOBase,
    bit_rate: int | None = None,
    num_channels: int | None = None,
    desired_sample_rate: int | None = None,
) -> None:
    """Encode audio samples to a file-like object.

    Args:
        samples: Audio samples tensor
        sample_rate: Sample rate in Hz
        format: Audio format (e.g., "wav", "mp3", "flac")
        file_like: File-like object that supports write() and seek() methods
        bit_rate: Optional bit rate for encoding
        num_channels: Optional number of output channels
        desired_sample_rate: Optional desired sample rate for the output.
    """
    assert _pybind_ops is not None

    if samples.dtype != torch.float32:
        raise ValueError(f"samples must have dtype torch.float32, got {samples.dtype}")

    _encode_audio_to_file_like(
        samples,
        sample_rate,
        format,
        _pybind_ops.create_file_like_context(file_like, True),  # True means for writing
        bit_rate,
        num_channels,
        desired_sample_rate,
    )


def encode_video_to_file_like(
    frames: torch.Tensor,
    frame_rate: float,
    format: str,
    file_like: io.RawIOBase | io.BufferedIOBase,
    codec: str | None = None,
    pixel_format: str | None = None,
    crf: int | float | None = None,
    preset: str | None = None,
    extra_options: list[str] | None = None,
) -> None:
    """Encode video frames to a file-like object.

    Args:
        frames: Video frames tensor. The device of the frames tensor will be used for encoding.
        frame_rate: Frame rate in frames per second
        format: Video format (e.g., "mp4", "mov", "mkv")
        file_like: File-like object that supports write() and seek() methods
        codec: Optional codec name (e.g., "libx264", "h264")
        pixel_format: Optional pixel format (e.g., "yuv420p", "yuv444p")
        crf: Optional constant rate factor for encoding quality
        preset: Optional encoder preset as string (e.g., "ultrafast", "medium")
        extra_options: Optional list of extra options as flattened key-value pairs
    """
    assert _pybind_ops is not None

    _encode_video_to_file_like(
        frames,
        frame_rate,
        format,
        _pybind_ops.create_file_like_context(file_like, True),  # True means for writing
        codec,
        pixel_format,
        crf,
        preset,
        extra_options,
    )


def get_frames_at_indices(
    decoder: torch.Tensor, *, frame_indices: torch.Tensor | list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(frame_indices, torch.Tensor):
        # Ensure indices is the correct dtype (int64)
        frame_indices = frame_indices.to(torch.int64)
    else:
        # Convert list to tensor for dispatch
        frame_indices = torch.tensor(frame_indices)
    return _get_frames_at_indices_tensor_input(decoder, frame_indices=frame_indices)


def get_frames_by_pts(
    decoder: torch.Tensor, *, timestamps: torch.Tensor | list[float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(timestamps, torch.Tensor):
        # Ensure indices is the correct dtype (float64)
        timestamps = timestamps.to(torch.float64)
    else:
        # Convert list to tensor for dispatch
        try:
            timestamps = torch.tensor(timestamps, dtype=torch.float64)
        except Exception as e:
            raise ValueError("Couldn't convert timestamps input to a tensor") from e
    return _get_frames_by_pts_tensor_input(decoder, timestamps=timestamps)


# ==============================
# Abstract impl for the operators. Needed by torch.compile.
# ==============================
@register_fake("torchcodec_ns::create_from_file")
def create_from_file_abstract(filename: str, seek_mode: str | None) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::_create_from_file_like")
def _create_from_file_like_abstract(
    file_like: int, seek_mode: str | None
) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::encode_audio_to_file")
def encode_audio_to_file_abstract(
    samples: torch.Tensor,
    sample_rate: int,
    filename: str,
    bit_rate: int | None = None,
    num_channels: int | None = None,
    desired_sample_rate: int | None = None,
) -> None:
    return


@register_fake("torchcodec_ns::encode_audio_to_tensor")
def encode_audio_to_tensor_abstract(
    samples: torch.Tensor,
    sample_rate: int,
    format: str,
    bit_rate: int | None = None,
    num_channels: int | None = None,
    desired_sample_rate: int | None = None,
) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::_encode_audio_to_file_like")
def _encode_audio_to_file_like_abstract(
    samples: torch.Tensor,
    sample_rate: int,
    format: str,
    file_like_context: int,
    bit_rate: int | None = None,
    num_channels: int | None = None,
    desired_sample_rate: int | None = None,
) -> None:
    return


@register_fake("torchcodec_ns::encode_video_to_file")
def encode_video_to_file_abstract(
    frames: torch.Tensor,
    frame_rate: float,
    filename: str,
    codec: str | None = None,
    pixel_format: str | None = None,
    preset: str | None = None,
    crf: int | float | None = None,
    extra_options: list[str] | None = None,
) -> None:
    return


@register_fake("torchcodec_ns::encode_video_to_tensor")
def encode_video_to_tensor_abstract(
    frames: torch.Tensor,
    frame_rate: float,
    format: str,
    codec: str | None = None,
    pixel_format: str | None = None,
    preset: str | None = None,
    crf: int | float | None = None,
    extra_options: list[str] | None = None,
) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::_encode_video_to_file_like")
def _encode_video_to_file_like_abstract(
    frames: torch.Tensor,
    frame_rate: float,
    format: str,
    file_like_context: int,
    codec: str | None = None,
    pixel_format: str | None = None,
    preset: str | None = None,
    crf: int | float | None = None,
    extra_options: list[str] | None = None,
) -> None:
    return


@register_fake("torchcodec_ns::create_from_tensor")
def create_from_tensor_abstract(
    video_tensor: torch.Tensor, seek_mode: str | None
) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::_add_video_stream")
def _add_video_stream_abstract(
    decoder: torch.Tensor,
    *,
    num_threads: int | None = None,
    dimension_order: str | None = None,
    stream_index: int | None = None,
    device: str = "cpu",
    device_variant: str = "ffmpeg",
    transform_specs: str = "",
    custom_frame_mappings: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    ) = None,
    color_conversion_library: str | None = None,
) -> None:
    return


@register_fake("torchcodec_ns::add_video_stream")
def add_video_stream_abstract(
    decoder: torch.Tensor,
    *,
    num_threads: int | None = None,
    dimension_order: str | None = None,
    stream_index: int | None = None,
    device: str = "cpu",
    device_variant: str = "ffmpeg",
    transform_specs: str = "",
    custom_frame_mappings: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    ) = None,
) -> None:
    return


@register_fake("torchcodec_ns::add_audio_stream")
def add_audio_stream_abstract(
    decoder: torch.Tensor,
    *,
    stream_index: int | None = None,
    sample_rate: int | None = None,
    num_channels: int | None = None,
) -> None:
    return


@register_fake("torchcodec_ns::seek_to_pts")
def seek_abstract(decoder: torch.Tensor, seconds: float) -> None:
    return


@register_fake("torchcodec_ns::get_next_frame")
def get_next_frame_abstract(
    decoder: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Images are 3 dimensions: height, width, channels.
    # The exact permutation depends on the constructor options passed in.
    image_size = [get_ctx().new_dynamic_size() for _ in range(3)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frame_at_pts")
def get_frame_at_pts_abstract(
    decoder: torch.Tensor, seconds: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(3)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frames_by_pts")
def get_frames_by_pts_abstract(
    decoder: torch.Tensor,
    *,
    timestamps: torch.Tensor | list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frame_at_index")
def get_frame_at_index_abstract(
    decoder: torch.Tensor, *, frame_index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(3)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frames_at_indices")
def get_frames_at_indices_abstract(
    decoder: torch.Tensor, *, frame_indices: torch.Tensor | list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frames_in_range")
def get_frames_in_range_abstract(
    decoder: torch.Tensor,
    *,
    start: int,
    stop: int,
    step: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frames_by_pts_in_range")
def get_frames_by_pts_in_range_abstract(
    decoder: torch.Tensor,
    *,
    start_seconds: float,
    stop_seconds: float,
    fps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frames_by_pts_in_range_audio")
def get_frames_by_pts_in_range_audio_abstract(
    decoder: torch.Tensor,
    *,
    start_seconds: float,
    stop_seconds: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return (torch.empty(image_size), torch.empty([], dtype=torch.float))


@register_fake("torchcodec_ns::_get_key_frame_indices")
def get_key_frame_indices_abstract(decoder: torch.Tensor) -> torch.Tensor:
    return torch.empty([], dtype=torch.int)


@register_fake("torchcodec_ns::get_json_metadata")
def get_json_metadata_abstract(decoder: torch.Tensor) -> str:
    return ""


@register_fake("torchcodec_ns::get_container_json_metadata")
def get_container_json_metadata_abstract(decoder: torch.Tensor) -> str:
    return ""


@register_fake("torchcodec_ns::get_stream_json_metadata")
def get_stream_json_metadata_abstract(decoder: torch.Tensor, stream_idx: int) -> str:
    return ""


@register_fake("torchcodec_ns::_test_frame_pts_equality")
def _test_frame_pts_equality_abstract(
    decoder: torch.Tensor,
    *,
    frame_index: int,
    pts_seconds_to_test: float,
) -> bool:
    return False


@register_fake("torchcodec_ns::_get_json_ffmpeg_library_versions")
def _get_json_ffmpeg_library_versions_abstract() -> str:
    return ""


@register_fake("torchcodec_ns::scan_all_streams_to_update_metadata")
def scan_all_streams_to_update_metadata_abstract(decoder: torch.Tensor) -> None:
    return


def get_ffmpeg_library_versions():
    versions_json = _get_json_ffmpeg_library_versions()
    return json.loads(versions_json)


@register_fake("torchcodec_ns::_get_backend_details")
def _get_backend_details_abstract(decoder: torch.Tensor) -> str:
    return ""
