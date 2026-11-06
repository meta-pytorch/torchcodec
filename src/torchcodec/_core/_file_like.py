# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expose Python file-like objects to the C++ FileLikeIO through ctypes.

See Note [File-like objects through ctypes] below.
"""

import ctypes
import operator
from typing import Any

# Note [File-like objects through ctypes]
#
# Decoders and encoders can read from or write to a Python file-like object.
# FFmpeg drives the IO from C++, so C++ has to call the object's read(), write()
# and seek() methods. This is how it does it without any of our compiled
# libraries using the CPython C API.
#
# The struct. _FileLikeCallbacks is a ctypes.Structure: each instance is an
# actual C struct in memory, laid out exactly like FileLikeCallbacks in
# FileLikeIO.h (an int64 id_ followed by five function pointers).
# create_file_like_context() fills one in and returns its address as a plain
# int, which goes through the custom ops as their `int file_like_context`
# argument. On the C++ side, adopt_file_like_context() casts that int back to a
# FileLikeCallbacks* and copies the struct into a FileLikeIO.
#
# The id_. The id_ field is the Python id() of the _Entry holding the file-like
# object, and its key in _entries. The callbacks are module-level functions
# shared by all file-like objects, so C++ passes id_ back as the first argument
# of every callback, and the callback looks up _entries[id_] to know which
# file-like object to use. C++ never dereferences it. id() is unique among
# objects that are alive at the same time, and an _Entry stays alive as long as
# it's in _entries, so two live entries never share an id_.
#
# The callbacks. ctypes turns each of the Python functions below into a C
# function pointer (_READ_FN_TYPE(_read), etc.): it generates, through libffi,
# a small piece of machine code that is a regular C function. FileLikeIO.cpp
# calls it like any other C function, e.g.
# `callbacks_.read(callbacks_.id_, buf, request)`. That machine code enters
# CPython's _ctypes module, which converts the C arguments to Python ints, calls
# _read(id_, buf, size), and converts its return value back to an int64.
# _read() calls the file-like object's read() and copies the bytes into buf.
#
# The GIL. The ops run with the GIL released, so C++ calls the callbacks from a
# thread that doesn't hold it. _ctypes wraps every callback in
# PyGILState_Ensure() / PyGILState_Release(), so the GIL is taken for the
# duration of each callback. On free-threaded builds there is no GIL, and
# PyGILState_Ensure() attaches the calling thread to the interpreter, which is
# what running Python code requires.
#
# Lifetime. The file-like object is kept alive by Python, not by C++: its _Entry
# sits in _entries, and C++ only holds its id_ and the function pointers. What
# C++ owns is the duty to release it: the FileLikeIO destructor calls
# release(id_), which removes the entry from _entries and with it the reference
# to the file-like object. FileLikeIO lives as long as the
# decoder or encoder that adopted it, so the file-like object stays alive until
# that decoder or encoder is destroyed. Every op adopts the context before doing
# anything else, so release() runs even if the op then fails. But if the address
# never reaches adopt_file_like_context(), e.g. because the op call fails before
# entering C++, release() is never called and the entry leaks, along with the
# file-like object. The function pointers themselves are module-level and live
# forever.
#
# Exceptions. A Python exception can't propagate through C code: ctypes would
# just print it and return to C++ as if nothing happened. So every callback
# catches everything, stores the exception on its _Entry, and returns
# _CALLBACK_ERROR. C++ then calls get_error() to fetch "TypeName: message" and
# raises a RuntimeError with it. The tradeoff is that the original exception
# type is lost: if a user's read() raises a TypeError, they get a RuntimeError
# whose message contains "TypeError: ...".
#
# Why not pybind11. pybind11, like any code using the CPython C API, is compiled
# into our library against the headers of one specific Python: it inlines
# reference counting, and relies on struct layouts and private symbols, all of
# which differ between Python versions and between regular and free-threaded
# builds. A library built that way only works with the Python it was built
# for. The limited API (abi3) would make it work across regular builds, but not
# free-threaded ones: those need abi3t, which only exists from Python 3.15. With
# ctypes we still go through the CPython C API, but that code lives in CPython's
# own _ctypes module, which is compiled for and ships with every Python. Our
# libraries only make plain C function calls, so they work with any Python
# version. The price is a fraction of a microsecond more per callback than
# pybind11, which doesn't show in practice: FFmpeg does IO in 64 KB chunks, so a
# typical op makes tens to hundreds of callbacks.

# Must match kFileLikeCallbackError in FileLikeIO.h.
_CALLBACK_ERROR = -(2**63)

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


class _Entry:
    __slots__ = ("file_like", "callbacks", "error")

    def __init__(self, file_like: Any):
        self.file_like = file_like
        self.callbacks: _FileLikeCallbacks | None = None
        self.error: BaseException | None = None


# Keeps the file-like objects alive until C++ calls release(). Keyed by id_.
_entries: dict[int, _Entry] = {}


# The callbacks catch all exceptions, see "Exceptions" in
# Note [File-like objects through ctypes].
def _read(id_: int, buf: int, size: int) -> int:
    entry = _entries[id_]
    try:
        data = entry.file_like.read(size)
        if not isinstance(data, bytes):
            raise TypeError(
                f"read() must return bytes, got {type(data).__name__} instead."
            )
        num_bytes = len(data)
        if num_bytes > size:
            raise ValueError(
                f"Requested up to {size} bytes but, received {num_bytes} bytes. "
                "The given object does not conform to read protocol of file object."
            )
        ctypes.memmove(buf, data, num_bytes)
        return num_bytes
    except BaseException as e:
        entry.error = e
        return _CALLBACK_ERROR


def _write(id_: int, buf: int, size: int) -> int:
    entry = _entries[id_]
    try:
        return operator.index(entry.file_like.write(ctypes.string_at(buf, size)))
    except BaseException as e:
        entry.error = e
        return _CALLBACK_ERROR


def _seek(id_: int, offset: int, whence: int) -> int:
    entry = _entries[id_]
    try:
        return operator.index(entry.file_like.seek(offset, whence))
    except BaseException as e:
        entry.error = e
        return _CALLBACK_ERROR


def _release(id_: int) -> None:
    _entries.pop(id_, None)


def _get_error(id_: int, buf: int, buf_len: int) -> int:
    entry = _entries[id_]
    error, entry.error = entry.error, None
    message = f"{type(error).__name__}: {error}".encode()
    # Don't truncate in the middle of a multi-byte UTF-8 sequence: the message
    # must remain valid UTF-8 to be turned back into a Python str.
    message = message[:buf_len].decode(errors="ignore").encode()
    ctypes.memmove(buf, message, len(message))
    return len(message)


# These must stay alive for as long as C++ may call them, i.e. forever.
_read_fn = _READ_FN_TYPE(_read)
_write_fn = _WRITE_FN_TYPE(_write)
_seek_fn = _SEEK_FN_TYPE(_seek)
_release_fn = _RELEASE_FN_TYPE(_release)
_get_error_fn = _GET_ERROR_FN_TYPE(_get_error)


def create_file_like_context(file_like: Any, is_for_writing: bool) -> int:
    """Return an int to pass as the ``file_like_context`` of a custom op.

    The returned value must be passed to exactly one custom op, which becomes
    responsible for releasing the file-like object. See "Lifetime" in
    Note [File-like objects through ctypes].
    """
    if is_for_writing:
        if not hasattr(file_like, "write"):
            raise RuntimeError(
                "File like object must implement a write method for writing."
            )
    elif not hasattr(file_like, "read"):
        raise RuntimeError("File like object must implement a read method for reading.")
    if not hasattr(file_like, "seek"):
        raise RuntimeError("File like object must implement a seek method.")

    entry = _Entry(file_like)
    id_ = id(entry)
    entry.callbacks = _FileLikeCallbacks(
        id_, _read_fn, _write_fn, _seek_fn, _release_fn, _get_error_fn
    )
    _entries[id_] = entry
    return ctypes.addressof(entry.callbacks)
