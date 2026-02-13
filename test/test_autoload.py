# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os


def test_autoload():
    switch = os.getenv("TORCHCODEC_DEVICE_BACKEND_AUTOLOAD", "0")

    # After importing the test extension, the value of this environment variable should be true
    is_imported = os.getenv("IS_CUSTOM_DEVICE_BACKEND_IMPORTED", "0")

    # Both values should be equal
    assert is_imported == switch
