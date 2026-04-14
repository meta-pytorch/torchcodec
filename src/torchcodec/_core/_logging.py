# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torchcodec._core.ops import _is_logging_enabled, _set_logging_enabled

_LG = logging.getLogger("torchcodec")
_LG.setLevel(logging.DEBUG)

_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("[torchcodec %(filename)s:%(lineno)d] %(message)s")
)


def set_logging_enabled(enabled: bool) -> None:
    """Enable or disable torchcodec logging (both Python and C++ sides)."""
    _set_logging_enabled(enabled)
    if enabled:
        if _handler not in _LG.handlers:
            _LG.addHandler(_handler)
    else:
        _LG.removeHandler(_handler)


def is_logging_enabled() -> bool:
    """Return whether torchcodec logging is currently enabled."""
    return _is_logging_enabled()
