# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expose Python file-like objects to the C++ FileLikeIO through ctypes.

See Note [File-like objects through ctypes] below.
"""

import ctypes
from dataclasses import dataclass
from typing import Any

# Note [File-like objects through ctypes]
#
# File-like objects are user-specified Python objects with a read, write, and
# seek method. They can be used as the source or destination of a decoder or
# encoder, instead of a file on disk.
#
# Their read(), write(), and seek() methods are called from C++ by FFmpeg. We do
# this binding through ctypes (not pybind11! See below why).
#
# When a user gives us a file-like object, we create a _FileLikeContext to hold
# it, and we associate it with a _FileLikeCallbacks struct, which mirrors the
# C++ FileLikeCallbacks struct. This _FileLikeCallbacks struct **is a C struct
# in memory!** (that's what ctypes does). It holds callback pointers to the
# Python functions below (_read, _write, _seek, _release, _get_error). Each of
# these functions then calls the corresponding method of the file-like object:
# they know how, because they're given the Python id() of the _FileLikeContext
# that they must call those methods on.
#
# ctypes automatically acquires the GIL for each callback.
#
# The _FileLikeContext object is kept alive by Python, not by C++. It
# is stored in _ALIVE_CONTEXTS until its release() callback is called from C++,
# upon destruction of its associated FileLikeIO object.
#
# A Python exception cannot be raised from within a callback because it cannot
# cross the C++ boundary. So all callbacks catch exceptions and return a special
# value instead (_CALLBACK_ERROR). When C++ sees that value, it calls the
# get_error() callback to fetch the error message, which is then turned into a
# RuntimeError via STD_TORCH_CHECK(false, the_message).
#
# Why go the hard way of using ctypes instead of just using pybind? Mainly
# because we want our wheels to be py3-none, i.e. we don't want our wheels to be
# tied to a CPython version. We could use pybind and rely on stable ABIs like
# abi3/abi3t, but only from 3.15, and we don't want to wait that long.
#
# Note that ctypes does call the same Python C APIs that pybind does, so why can
# we keep our wheels py3-none?  The main difference here is that *our* C++
# doesn't use them (directly or transitively through pybind). We only rely on
# ctypes, from Python, and C++ has no idea it's talking to Python.  ctypes
# itself *is* compiled against a specific CPython version, just like pybind code
# would be. But that's fine: ctypes is part of CPython's standard library, so
# every Python installation ships its own ctypes, compiled for that exact
# version.

# Must match kFileLikeCallbackError in FileLikeIO.h.
_CALLBACK_ERROR = -(2**63)
_INT64_MAX = 2**63 - 1

_READ_FN_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64
)
_WRITE_FN_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64
)
_SEEK_FN_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
)
_RELEASE_FN_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int64)
_GET_ERROR_FN_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64
)


# Must match the layout of FileLikeCallbacks in FileLikeIO.h.
class _FileLikeCallbacks(ctypes.Structure):
    _fields_ = [
        ("id_", ctypes.c_int64),
        ("read", _READ_FN_TYPE),
        ("write", _WRITE_FN_TYPE),
        ("seek", _SEEK_FN_TYPE),
        ("release", _RELEASE_FN_TYPE),
        ("get_error", _GET_ERROR_FN_TYPE),
    ]


@dataclass
class _FileLikeContext:
    file_like: Any
    callbacks: _FileLikeCallbacks | None = None
    error_message: str | None = None


# Keeps the file-like objects alive until C++ calls release(). Keyed by id_.
_ALIVE_CONTEXTS: dict[int, _FileLikeContext] = {}


def _format_error(e: BaseException) -> str:
    try:
        return f"{type(e).__name__}: {e}"
    except BaseException:
        # e.g. the exception's __str__ raised.
        return "<unprintable exception>"


# NEVER call this outside of a try/except if within a callback
def _check_int64_result(result: Any, method: str) -> int:
    # ctypes would turn a non-int, e.g. None, into a garbage value.
    if not isinstance(result, int):
        raise TypeError(f"{method}() must return an int, got {type(result).__name__}.")
    # Excludes _CALLBACK_ERROR, which C++ would take for an error. Also excludes
    # values outside the int64 range, which ctypes would silently wrap, e.g.
    # 2**63 would reach C++ as _CALLBACK_ERROR.
    if not _CALLBACK_ERROR < result <= _INT64_MAX:
        raise ValueError(
            f"{method}() returned {result}, which is outside the supported range "
            f"[{_CALLBACK_ERROR + 1}, {_INT64_MAX}]."
        )
    return result


def _read(id_: int, buf: int, size: int) -> int:
    context = _ALIVE_CONTEXTS[id_]
    try:
        # Fill buf unless we reach EOF: read() may return fewer bytes than
        # requested.
        total_num_read = 0
        while total_num_read < size:
            request = size - total_num_read
            data = context.file_like.read(request)
            if not isinstance(data, bytes):
                raise TypeError(
                    f"read() must return bytes, got {type(data).__name__} instead."
                )
            num_bytes = len(data)
            if num_bytes == 0:
                break
            # Guards the memmove below against overflowing buf.
            if num_bytes > request:
                raise ValueError(
                    f"Requested up to {request} bytes but, received {num_bytes} "
                    "bytes. The given object does not conform to read protocol "
                    "of file object."
                )
            ctypes.memmove(buf + total_num_read, data, num_bytes)
            total_num_read += num_bytes
        return total_num_read
    except BaseException as e:
        context.error_message = _format_error(e)
        return _CALLBACK_ERROR


def _write(id_: int, buf: int, size: int) -> int:
    context = _ALIVE_CONTEXTS[id_]
    try:
        result = context.file_like.write(ctypes.string_at(buf, size))
        return _check_int64_result(result, "write")
    except BaseException as e:
        context.error_message = _format_error(e)
        return _CALLBACK_ERROR


def _seek(id_: int, offset: int, whence: int) -> int:
    context = _ALIVE_CONTEXTS[id_]
    try:
        result = context.file_like.seek(offset, whence)
        return _check_int64_result(result, "seek")
    except BaseException as e:
        context.error_message = _format_error(e)
        return _CALLBACK_ERROR


def _release(id_: int) -> None:
    _ALIVE_CONTEXTS.pop(id_, None)


def _get_error(id_: int, buf: int, buf_len: int) -> int:
    try:
        context = _ALIVE_CONTEXTS[id_]
        message, context.error_message = context.error_message, None
        encoded = (message or "unknown error").encode(errors="replace")
        # Don't truncate in the middle of a multi-byte UTF-8 sequence: the
        # message must remain valid UTF-8 to be turned back into a Python str.
        encoded = encoded[:buf_len].decode(errors="ignore").encode()
        ctypes.memmove(buf, encoded, len(encoded))
        return len(encoded)
    except BaseException:
        return 0


# These must stay alive for as long as C++ may call them, i.e. forever.
_read_fn = _READ_FN_TYPE(_read)
_write_fn = _WRITE_FN_TYPE(_write)
_seek_fn = _SEEK_FN_TYPE(_seek)
_release_fn = _RELEASE_FN_TYPE(_release)
_get_error_fn = _GET_ERROR_FN_TYPE(_get_error)


def create_file_like_context(file_like: Any, is_for_writing: bool) -> int:
    """Create a _FileLikeContext and return the C address of its _FileLikeCallbacks struct."""
    if is_for_writing:
        if not hasattr(file_like, "write"):
            raise RuntimeError(
                "File like object must implement a write method for writing."
            )
    elif not hasattr(file_like, "read"):
        raise RuntimeError("File like object must implement a read method for reading.")
    if not hasattr(file_like, "seek"):
        raise RuntimeError("File like object must implement a seek method.")

    context = _FileLikeContext(file_like)
    id_ = id(context)
    context.callbacks = _FileLikeCallbacks(
        id_, _read_fn, _write_fn, _seek_fn, _release_fn, _get_error_fn
    )
    _ALIVE_CONTEXTS[id_] = context
    return ctypes.addressof(context.callbacks)
