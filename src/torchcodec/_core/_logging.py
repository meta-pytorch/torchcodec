# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torchcodec._core.ops import _is_logging_enabled, _set_logging_enabled

_LG = logging.getLogger("torchcodec")

# Add a stderr handler so logs are visible when logging is enabled.
if not _LG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[torchcodec %(filename)s:%(lineno)d] %(message)s")
    )
    _LG.addHandler(_handler)

# Logging is disabled by default. The Python logger level is set high enough
# to suppress all messages; it gets lowered when the user enables logging.
_LG.setLevel(logging.CRITICAL + 1)


def set_logging_enabled(enabled: bool) -> None:
    """Enable or disable torchcodec logging (both Python and C++ sides)."""
    _set_logging_enabled(enabled)
    if enabled:
        _LG.setLevel(logging.DEBUG)
    else:
        _LG.setLevel(logging.CRITICAL + 1)


def is_logging_enabled() -> bool:
    """Return whether torchcodec logging is currently enabled."""
    return _is_logging_enabled()
