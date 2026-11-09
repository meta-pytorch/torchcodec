# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assert that using torchcodec on a free-threaded Python keeps the GIL disabled.

Importing any C extension that doesn't declare free-threading support silently
re-enables the GIL for the whole process (with only a RuntimeWarning). This runs
the smoke tests in-process, then checks the GIL is still off.

Must be run from the repo root, on a free-threaded Python, with FFmpeg
installed. Extra arguments are forwarded to pytest.

Usage:
    python packaging/assert_gil_disabled.py [PYTEST_ARGS...]
"""

import sys
import sysconfig

import pytest


def main() -> None:
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        sys.exit(f"Not a free-threaded Python build: {sys.version}")
    if sys.flags.gil is not None:
        sys.exit("The GIL is forced on or off by PYTHON_GIL or -X gil, unset it.")

    from torchcodec._core import ops

    # The FFmpeg-dependent modules are only imported when FFmpeg is found.
    assert ops._FFMPEG_AVAILABLE, "FFmpeg wasn't loaded, this check is vacuous"

    exit_code = pytest.main(["test/smoke_test.py", *sys.argv[1:]])
    if exit_code != 0:
        sys.exit(f"Smoke tests failed with exit code {exit_code}")

    if sys._is_gil_enabled():
        sys.exit(
            "The GIL was re-enabled! Some imported C extension doesn't declare "
            "free-threading support; re-run with `-W error::RuntimeWarning` to "
            "see which one."
        )
    print("OK: the GIL is still disabled")


if __name__ == "__main__":
    main()
