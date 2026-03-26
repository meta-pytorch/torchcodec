# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

_LG = logging.getLogger("torchcodec")

# Default to showing all logs. No level filtering for now.
_LG.setLevel(logging.DEBUG)

# Add a stderr handler so logs are visible without user configuration.
if not _LG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[torchcodec %(filename)s:%(lineno)d] %(message)s")
    )
    _LG.addHandler(_handler)
