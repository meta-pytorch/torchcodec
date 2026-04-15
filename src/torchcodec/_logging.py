# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import logging
from typing import Literal

from torchcodec._core.ops import _get_log_level, _set_cpp_log_level

_LG = logging.getLogger("torchcodec")
_LG.setLevel(logging.DEBUG)

_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("[torchcodec %(filename)s:%(lineno)d] %(message)s")
)


# Keep in sync with LogLevel in torchcodec/_core/Logging.h.
class _LogLevel(enum.Enum):
    OFF = 0
    ALL = 1


def set_log_level(level: Literal["OFF", "ALL"]) -> None:
    """Set the torchcodec log level.

    Supported levels: "OFF", "ALL".
    """
    level = level.upper()
    try:
        log_level = _LogLevel[level]
    except KeyError:
        raise ValueError(
            f"Invalid log level: {level!r}. Supported levels: {[lvl.name for lvl in _LogLevel]}"
        )
    _set_cpp_log_level(log_level.value)
    # Probably not thread-safe, and probably OK still.
    if log_level != _LogLevel.OFF:
        if _handler not in _LG.handlers:
            _LG.addHandler(_handler)
    else:
        _LG.removeHandler(_handler)


def get_log_level() -> str:
    """Return the current torchcodec log level as a string."""
    level_int = _get_log_level()
    try:
        return _LogLevel(level_int).name
    except ValueError:
        raise RuntimeError(
            f"C++ log level {level_int} has no matching Python level name. "
        )
