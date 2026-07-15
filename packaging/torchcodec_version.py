# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""scikit-build-core dynamic-metadata provider for torchcodec's version.

This restores the version-handling logic that used to live in setup.py's
``_write_version_files()`` before the migration to scikit-build-core. It is
wired up from pyproject.toml via::

    [tool.scikit-build.metadata.version]
    provider = "torchcodec_version"
    provider-path = "packaging"

scikit-build-core adds ``provider-path`` to ``sys.path`` and imports this
module to resolve the (dynamic) ``version`` metadata field. The resolved value
is also what gets substituted into the generated ``version.py`` files (see the
``[[tool.scikit-build.generate]]`` sections in pyproject.toml).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

# packaging/ lives at the repo root, next to version.txt and .git.
_ROOT_DIR = Path(__file__).parent.parent.resolve()

__all__ = ["dynamic_metadata"]


def dynamic_metadata(field: str, settings: Mapping[str, Any]) -> str:
    assert field == "version"

    if version := os.getenv("BUILD_VERSION"):
        # BUILD_VERSION is set by the `test-infra` build jobs. It typically is
        # the content of `version.txt` plus some suffix like "+cpu" or "+cu112".
        # See
        # https://github.com/pytorch/test-infra/blob/61e6da7a6557152eb9879e461a26ad667c15f0fd/tools/pkg-helpers/pytorch_pkg_helpers/version.py#L113
        return version

    with open(_ROOT_DIR / "version.txt") as f:
        version = f.readline().strip()
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(_ROOT_DIR)
            )
            .decode("ascii")
            .strip()
        )
        version += "+" + sha[:7]
    except Exception:
        print("INFO: Didn't find sha. Is this a git repo?")

    return version
