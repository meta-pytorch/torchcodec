# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Torch-free implementation of the core decoder ops, dispatched to from ops.py
# when torch is not installed. Frames come back from the pybind frontend as
# DLPack capsules and are returned to the caller as numpy arrays (zero-copy).
#
# Only the video-decoding read path is supported torch-free; audio decoding,
# encoding, WAV, NVDEC cache, file-like / bytes / tensor sources, and
# torch.compile all require torch and raise a clear error here.

import json

import numpy as np

from torchcodec._core.ops import _pybind_ops


def _requires_torch(name):
    def _stub(*args, **kwargs):
        raise NotImplementedError(
            f"torchcodec.{name} requires PyTorch, which is not installed. "
            "Install torch to use this functionality. The torch-free build "
            "currently supports video decoding to numpy only."
        )

    return _stub


_DLPACK_CPU = 1
_DLPACK_CUDA = 2


def _cupy():
    try:
        import cupy  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "Decoding on CUDA without PyTorch returns a cupy array, but cupy "
            "is not installed. Install cupy (e.g. 'pip install cupy-cuda12x'), "
            "or install torch."
        ) from e
    return cupy


def _to_array(frame):
    """Convert a self-describing _DLPackFrame to its natural array type: numpy
    for CPU frames, cupy for CUDA frames (both zero-copy via DLPack). torch is
    not involved on this path."""
    device_type, _ = frame.__dlpack_device__()
    if device_type == _DLPACK_CUDA:
        return _cupy().from_dlpack(frame)
    return np.from_dlpack(frame)


# Backwards-compatible alias (CPU/GPU aware).
_to_numpy = _to_array


# ---- Construction ----------------------------------------------------------
def create_from_file(filename, seek_mode=None):
    # seek_mode is honored by torch path; the torch-free pybind path uses
    # approximate seeking + an explicit scan for content-based metadata.
    return _pybind_ops.create_decoder(filename)


create_from_tensor = _requires_torch("create_from_tensor")
create_from_bytes = _requires_torch("create_from_bytes")
create_from_file_like = _requires_torch("create_from_file_like")


def add_video_stream(
    decoder,
    *,
    num_threads=None,
    dimension_order=None,
    stream_index=None,
    device="cpu",
    device_variant="default",
    transform_specs="",
    custom_frame_mappings=None,
    output_dtype="uint8",
):
    # CPU and CUDA are supported torch-free (CUDA frames come back as cupy via
    # DLPack). Other device types still require torch.
    device_str = str(device)
    if not (device_str == "cpu" or device_str.startswith("cuda")):
        raise NotImplementedError(
            f"Device {device_str!r} requires PyTorch; the torch-free build "
            "supports CPU and CUDA only."
        )
    if output_dtype not in ("uint8", None):
        raise NotImplementedError(
            "Only uint8 output is supported in the torch-free build; "
            f"got output_dtype={output_dtype!r}."
        )
    if transform_specs:
        raise NotImplementedError("Transforms require PyTorch.")
    if custom_frame_mappings is not None:
        raise NotImplementedError("custom_frame_mappings requires PyTorch.")
    _pybind_ops.add_video_stream(
        decoder,
        dimension_order or "NCHW",
        -1 if stream_index is None else stream_index,
        num_threads,
        device_str,
        device_variant,
    )


add_audio_stream = _requires_torch("add_audio_stream")


def scan_all_streams_to_update_metadata(decoder):
    _pybind_ops.scan_all_streams(decoder)


seek_to_pts = _requires_torch("seek_to_pts")


# ---- Frame reads -----------------------------------------------------------
def get_next_frame(decoder):
    data, pts, duration = _pybind_ops.get_next_frame(decoder)
    return _to_numpy(data), np.float64(pts), np.float64(duration)


def get_frame_at_index(decoder, *, frame_index):
    data, pts, duration = _pybind_ops.get_frame_at_index(decoder, frame_index)
    return _to_numpy(data), np.float64(pts), np.float64(duration)


def get_frame_at_pts(decoder, seconds):
    data, pts, duration = _pybind_ops.get_frame_played_at(decoder, seconds)
    return _to_numpy(data), np.float64(pts), np.float64(duration)


def get_frames_in_range(decoder, *, start, stop, step=None):
    data, pts, duration = _pybind_ops.get_frames_in_range(
        decoder, start, stop, 1 if step is None else step
    )
    return _to_numpy(data), _to_numpy(pts), _to_numpy(duration)


def get_frames_at_indices(decoder, *, frame_indices):
    indices = [int(i) for i in frame_indices]
    data, pts, duration = _pybind_ops.get_frames_at_indices(decoder, indices)
    return _to_numpy(data), _to_numpy(pts), _to_numpy(duration)


def get_frames_by_pts(decoder, *, timestamps):
    seconds = [float(t) for t in timestamps]
    data, pts, duration = _pybind_ops.get_frames_by_pts(decoder, seconds)
    return _to_numpy(data), _to_numpy(pts), _to_numpy(duration)


def get_frames_by_pts_in_range(decoder, *, start_seconds, stop_seconds, fps=None):
    data, pts, duration = _pybind_ops.get_frames_by_pts_in_range(
        decoder, start_seconds, stop_seconds, fps
    )
    return _to_numpy(data), _to_numpy(pts), _to_numpy(duration)


get_frames_by_pts_in_range_audio = _requires_torch("get_frames_by_pts_in_range_audio")


def _get_key_frame_indices(decoder):
    return _to_numpy(_pybind_ops.get_key_frame_indices(decoder))


# ---- Metadata --------------------------------------------------------------
def get_json_metadata(decoder):
    return _pybind_ops.get_json_metadata(decoder)


def _get_container_json_metadata(decoder):
    return _pybind_ops.get_container_json_metadata(decoder)


def _get_stream_json_metadata(decoder, stream_index):
    return _pybind_ops.get_stream_json_metadata(decoder, stream_index)


def _get_backend_details(decoder):
    # CPU-only torch-free build has no CUDA fallback details to report.
    return ""


def _test_frame_pts_equality(decoder, *, frame_index, pts_seconds_to_test):
    raise NotImplementedError("_test_frame_pts_equality requires PyTorch.")


def get_ffmpeg_library_versions():
    # Not exposed torch-free yet; return empty mapping rather than failing
    # import-time consumers.
    return {}


# ---- Encoding / WAV / NVDEC (torch only) -----------------------------------
create_streaming_encoder = _requires_torch("create_streaming_encoder")
streaming_encoder_close = _requires_torch("streaming_encoder_close")
streaming_encoder_add_video_stream = _requires_torch(
    "streaming_encoder_add_video_stream"
)
streaming_encoder_add_audio_stream = _requires_torch(
    "streaming_encoder_add_audio_stream"
)
streaming_encoder_open_file = _requires_torch("streaming_encoder_open_file")
streaming_encoder_open_file_like = _requires_torch("streaming_encoder_open_file_like")
streaming_encoder_add_frames = _requires_torch("streaming_encoder_add_frames")
streaming_encoder_add_samples = _requires_torch("streaming_encoder_add_samples")

create_wav_decoder_from_file = _requires_torch("create_wav_decoder_from_file")
create_wav_decoder_from_tensor = _requires_torch("create_wav_decoder_from_tensor")
create_wav_decoder_from_bytes = _requires_torch("create_wav_decoder_from_bytes")
create_wav_decoder_from_file_like = _requires_torch("create_wav_decoder_from_file_like")
get_wav_samples_in_range = _requires_torch("get_wav_samples_in_range")
get_wav_metadata_from_decoder = _requires_torch("get_wav_metadata_from_decoder")

get_nvdec_cache_capacity = _requires_torch("get_nvdec_cache_capacity")
set_nvdec_cache_capacity = _requires_torch("set_nvdec_cache_capacity")
_get_nvdec_cache_size = _requires_torch("_get_nvdec_cache_size")


# Logging control is not exposed torch-free yet; benign no-ops so the logging
# module imports and basic logging still works.
def _set_cpp_log_level(level):
    return None


def _get_log_level():
    return 0


# json is used by callers that parse get_ffmpeg_library_versions; keep imported.
_ = json
