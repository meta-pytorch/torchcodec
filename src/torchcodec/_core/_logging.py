# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torchcodec._core.ops import _get_log_level, _set_log_level

_LG = logging.getLogger("torchcodec")
_LG.setLevel(logging.DEBUG)

_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("[torchcodec %(filename)s:%(lineno)d] %(message)s")
)

# Maps user-facing level names to C++ log level ints.
# Currently only OFF and ALL are supported; more levels will be added later.
_LEVEL_NAME_TO_INT = {
    "OFF": 0,
    "ALL": 1,
}

_LEVEL_INT_TO_NAME = {v: k for k, v in _LEVEL_NAME_TO_INT.items()}


def set_log_level(level: str) -> None:
    """Set the torchcodec log level (both Python and C++ sides).

    Supported levels: "OFF", "ALL".
    """
    level = level.upper()
    if level not in _LEVEL_NAME_TO_INT:
        raise ValueError(
            f"Invalid log level: {level!r}. Supported levels: {list(_LEVEL_NAME_TO_INT.keys())}"
        )
    level_int = _LEVEL_NAME_TO_INT[level]
    _set_log_level(level_int)
    if level_int > 0:
        if _handler not in _LG.handlers:
            _LG.addHandler(_handler)
    else:
        _LG.removeHandler(_handler)


def get_log_level() -> str:
    """Return the current torchcodec log level as a string."""
    level_int = _get_log_level()
    return _LEVEL_INT_TO_NAME.get(level_int, f"UNKNOWN({level_int})")
