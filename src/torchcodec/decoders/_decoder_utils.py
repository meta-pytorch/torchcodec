# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import contextvars
from collections.abc import Generator
from contextlib import contextmanager

from torchcodec import _core as core


# Thread-local and async-safe storage for the current CUDA backend
_CUDA_BACKEND: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_CUDA_BACKEND", default="ffmpeg"
)


@contextmanager
def set_cuda_backend(backend: str) -> Generator[None, None, None]:
    """Context Manager to set the CUDA backend for :class:`~torchcodec.decoders.VideoDecoder`.

    This context manager allows you to specify which CUDA backend implementation
    to use when creating :class:`~torchcodec.decoders.VideoDecoder` instances
    with CUDA devices.

    .. note::
        **We recommend trying the "beta" backend instead of the default "ffmpeg"
        backend!** The beta backend is faster, and will eventually become the
        default in future versions. It may have rough edges that we'll polish
        over time, but it's already quite stable and ready for adoption. Let us
        know what you think!

    Only the creation of the decoder needs to be inside the context manager, the
    decoding methods can be called outside of it. You still need to pass
    ``device="cuda"`` when creating the
    :class:`~torchcodec.decoders.VideoDecoder` instance. If a CUDA device isn't
    specified, this context manager will have no effect. See example below.

    This is thread-safe and async-safe.

    Args:
        backend (str): The CUDA backend to use. Can be "ffmpeg" (default) or
            "beta". We recommend trying "beta" as it's faster!

    Example:
        >>> with set_cuda_backend("beta"):
        ...     decoder = VideoDecoder("video.mp4", device="cuda")
        ...
        ... # Only the decoder creation needs to be part of the context manager.
        ... # Decoder will now the beta CUDA implementation:
        ... decoder.get_frame_at(0)
    """
    backend = backend.lower()
    if backend not in ("ffmpeg", "beta"):
        raise ValueError(
            f"Invalid CUDA backend ({backend}). Supported values are 'ffmpeg' and 'beta'."
        )

    previous_state = _CUDA_BACKEND.set(backend)
    try:
        yield
    finally:
        _CUDA_BACKEND.reset(previous_state)


def _get_cuda_backend() -> str:
    return _CUDA_BACKEND.get()


def set_nvdec_cache_capacity(capacity: int) -> None:
    """Set the maximum number of NVDEC decoders cached per GPU device.

    The NVDEC decoder cache stores hardware decoders for reuse, avoiding the
    overhead of creating new decoders for subsequent video decoding operations on
    the same GPU. This function sets the capacity of the cache, i.e. the maximum
    number of decoders that can be cached per device. The default capacity is 20
    decoders per device.

    Args:
        capacity (int): The maximum number of NVDEC decoders to cache per GPU
            device. Must be non-negative. Setting to 0 disables caching.
    """
    core.set_nvdec_cache_capacity(capacity)


def get_nvdec_cache_capacity() -> int:
    """Get the capacity of the per-device NVDEC decoder cache.

    Returns:
        int: The maximum number of NVDEC decoders that can be cached per GPU
            device.
    """
    return core.get_nvdec_cache_capacity()
