# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Output "bridge": selects the array type that decoders return.

Inspired by decord's bridge. The bridge controls only the OUTPUT type; it does
not change how decoding executes (that is determined by whether torch is
installed). Conversions are zero-copy via DLPack / buffer protocol.

- Default is "torch" when torch is installed, else "numpy".
- ``set_bridge("torch")`` requires torch to be installed.
- Converting a CUDA frame to "numpy" is an error (numpy is host-only).
"""

import contextvars

# Neither torch nor numpy is required to be installed; torchcodec needs at least
# one. We import both lazily-guarded so a torch-only user doesn't need numpy and
# vice versa.
try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


_VALID_BRIDGES = ("torch", "numpy")
_DEFAULT_BRIDGE = "torch" if _HAS_TORCH else "numpy"
_BRIDGE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "torchcodec_bridge", default=_DEFAULT_BRIDGE
)


def set_bridge(bridge: str) -> None:
    """Set the array type returned by decoders.

    Args:
        bridge (str): ``"torch"`` (return ``torch.Tensor``) or ``"numpy"``
            (return ``numpy.ndarray``). The default is ``"torch"`` when PyTorch
            is installed, otherwise ``"numpy"``. This is thread-safe and
            async-safe (it uses a context variable).
    """
    bridge = bridge.lower()
    if bridge not in _VALID_BRIDGES:
        raise ValueError(
            f"Invalid bridge ({bridge!r}). Supported values are "
            f"{', '.join(_VALID_BRIDGES)}."
        )
    if bridge == "torch" and not _HAS_TORCH:
        raise RuntimeError(
            "set_bridge('torch') requires PyTorch, which is not installed."
        )
    if bridge == "numpy" and not _HAS_NUMPY:
        raise RuntimeError(
            "set_bridge('numpy') requires numpy, which is not installed."
        )
    _BRIDGE.set(bridge)


def get_bridge() -> str:
    """Return the current bridge (``"torch"`` or ``"numpy"``)."""
    return _BRIDGE.get()


def _to_torch(array):
    if _HAS_TORCH and isinstance(array, torch.Tensor):
        return array
    if _HAS_NUMPY and isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    # DLPack-capable object / capsule.
    return torch.from_dlpack(array)


def _to_numpy(array):
    if not _HAS_NUMPY:
        raise RuntimeError(
            "Returning numpy arrays requires numpy, which is not installed."
        )
    if isinstance(array, np.ndarray):
        return array
    if _HAS_TORCH and isinstance(array, torch.Tensor):
        if array.is_cuda:
            raise RuntimeError(
                "Cannot return a CUDA frame as a numpy array (the 'numpy' "
                "bridge is host-only). Use the 'torch' bridge, or decode on a "
                "CPU device."
            )
        return array.numpy()
    # DLPack-capable object / capsule (np.from_dlpack rejects non-CPU).
    return np.from_dlpack(array)


def to_bridge_array(array):
    """Convert a decoded array (torch tensor / numpy / DLPack) to the current
    bridge's type. No-op in the common case (output already matches)."""
    if get_bridge() == "torch":
        return _to_torch(array)
    return _to_numpy(array)
