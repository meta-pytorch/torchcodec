# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tre

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _test_autoload(tmp_path, enable_autoload=True):
    test_directory = Path(__file__).parent

    # Build the test plugin
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--root",
        "./install",
        test_directory / "plugin",
    ]
    return_code = subprocess.run(cmd, cwd=tmp_path, env=os.environ)
    assert return_code.returncode == 0

    # "install" the test modules and run tests
    python_path = os.environ.get("PYTHONPATH", "")
    torchcodec_autoload = os.environ.get("PYTHONPATH", "")

    try:
        install_directory = ""

        # install directory is the one that is named site-packages
        for path in (tmp_path / "install").rglob("*"):
            if path.is_dir() and "-packages" in path.name:
                install_directory = str(path)

        print(f">>>>> !!!! install_directory={install_directory}")
        assert install_directory, "install_directory must not be empty"
        os.environ["PYTHONPATH"] = os.pathsep.join([install_directory, python_path])
        os.environ["TORCHCODEC_DEVICE_BACKEND_AUTOLOAD"] = str(int(enable_autoload))

        cmd = [sys.executable, "-m", "pytest", "test_autoload.py"]
        return_code = subprocess.run(cmd, cwd=Path(__file__).parent, env=os.environ)
        assert return_code.returncode == 0
    finally:
        os.environ["PYTHONPATH"] = python_path
        if torchcodec_autoload != "":
            os.environ.pop("TORCHCODEC_DEVICE_BACKEND_AUTOLOAD")


@pytest.mark.parametrize("enable_autoload", [True, False])
def test_plugin_autoload(tmp_path, enable_autoload):
    return _test_autoload(tmp_path, enable_autoload=enable_autoload)
