# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


def _process_local(hint: str):
    """Class decorator refusing to pickle, copy or deepcopy the instances.

    These objects carry a pointer to an FFmpeg object as their payload: the
    handle tensor's *data* is the address (see ``wrap_pointer_to_tensor`` in
    ``custom_ops.cpp``). Pickle would copy those 8 bytes into another process
    without complaining, and nothing validates a handle before dereferencing
    it, so the result would be a wild pointer rather than an error. ``copy``
    and ``deepcopy`` are just as wrong: the copy holds the same address without
    holding the deleter that keeps the FFmpeg object alive.

    Overriding ``__reduce__`` covers all of those, and ``torch.save`` too.
    """

    def decorator(cls):
        def __reduce__(self):
            raise TypeError(
                f"{type(self).__name__} cannot be pickled, copied or sent to "
                f"another process: it is a handle to an FFmpeg object that "
                f"only exists in this one. {hint}"
            )

        cls.__reduce__ = __reduce__
        return cls

    return decorator
