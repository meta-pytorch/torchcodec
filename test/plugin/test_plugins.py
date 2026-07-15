# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("enable_autoload", [True, False])
def test_plugin_autoload(tmp_path, enable_autoload):
    plugin_dir = Path(__file__).parent
    install_dir = str(tmp_path / "install")

    # Install the dummy plugin into a temp directory. We use --target so that
    # the .dist-info metadata and the plugin package are installed flat into a
    # separate directory to avoid polluting the current env's site-packages.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--target", install_dir, plugin_dir],
        check=True,
    )

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([install_dir, os.environ.get("PYTHONPATH", "")]),
        "TORCHCODEC_DEVICE_BACKEND_AUTOLOAD": str(int(enable_autoload)),
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import torchcodec

loaded = os.environ.get("IS_DUMMY_PLUGIN_LOADED") == "1"
autoload = os.environ.get("TORCHCODEC_DEVICE_BACKEND_AUTOLOAD") == "1"
assert loaded == autoload, f"{loaded=} {autoload=}"
""",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
