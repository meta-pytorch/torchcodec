# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assert the FFmpeg major version that torchcodec loads at runtime.

A conda solve (e.g. installing libheif) can silently swap the installed FFmpeg
onto a major version different from the one we asked for.
Usage:
    python packaging/assert_ffmpeg_version.py 5
"""

import sys

import torchcodec


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: assert_ffmpeg_version.py EXPECTED_MAJOR")

    expected_major = int(sys.argv[1])
    actual_major = torchcodec.ffmpeg_major_version

    if actual_major != expected_major:
        sys.exit(
            f"FFmpeg major version mismatch! Expected {expected_major}, but "
            f"torchcodec loaded FFmpeg {actual_major}. A conda solve likely "
            f"swapped it."
        )

    print(f"FFmpeg major version OK: {actual_major}.")


if __name__ == "__main__":
    main()
