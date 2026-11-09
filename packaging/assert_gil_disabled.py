# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assert that using torchcodec on a free-threaded Python keeps the GIL disabled.

Importing any C extension that doesn't declare free-threading support silently
re-enables the GIL for the whole process (with only a RuntimeWarning). This
imports torchcodec and exercises its main entry points, then checks the GIL is
still off.

Must be run from the repo root, on a free-threaded Python, with FFmpeg
installed.

Usage:
    python packaging/assert_gil_disabled.py
"""

import io
import sys
import sysconfig
from pathlib import Path

RESOURCES = Path("test/resources")


def main() -> None:
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        sys.exit(f"Not a free-threaded Python build: {sys.version}")
    if sys.flags.gil is not None:
        sys.exit("The GIL is forced on or off by PYTHON_GIL or -X gil, unset it.")

    import torchcodec
    from torchcodec._core import ops
    from torchcodec.decoders import AudioDecoder, decode_jpeg, VideoDecoder
    from torchcodec.encoders import AudioEncoder

    # The FFmpeg-dependent modules are only imported when FFmpeg is found.
    assert ops._FFMPEG_AVAILABLE, "FFmpeg wasn't loaded, this check is vacuous"
    print(f"{torchcodec.__version__ = }, {torchcodec.ffmpeg_major_version = }")

    video_path = RESOURCES / "nasa_13013.mp4"
    VideoDecoder(video_path)[0]
    VideoDecoder(io.BytesIO(video_path.read_bytes()))[0]
    samples = AudioDecoder(RESOURCES / "sine_mono_s32.wav").get_all_samples()
    AudioEncoder(samples.data, sample_rate=samples.sample_rate).to_tensor("mp3")
    decode_jpeg(RESOURCES / "gradient.jpg")

    if sys._is_gil_enabled():
        sys.exit(
            "The GIL was re-enabled! Some imported C extension doesn't declare "
            "free-threading support; re-run with `-W error::RuntimeWarning` to "
            "see which one."
        )
    print("OK: the GIL is still disabled")


if __name__ == "__main__":
    main()
