# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import subprocess
import sys
import textwrap

from .utils import needs_cuda


class TestLogging:
    @needs_cuda
    def test_logging_cuda_fallback(self):
        """Test that both C++ and Python logs fire for CUDA fallback,
        and that set_log_level("OFF") silences them."""
        script = textwrap.dedent(
            """\
            from torchcodec._logging import set_log_level
            set_log_level("ALL")
            from torchcodec.decoders import VideoDecoder, set_cuda_backend
            from test.utils import H265_VIDEO
            with set_cuda_backend("beta"):
                decoder = VideoDecoder(H265_VIDEO.path, device="cuda")
            # TODO: Remove this line and the assert below once
            # the Python-side fallback log in _video_decoder.py is removed.
            _ = decoder.cpu_fallback

            set_log_level("OFF")
            with set_cuda_backend("beta"):
                decoder2 = VideoDecoder(H265_VIDEO.path, device="cuda")
        """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        expected = (
            r"\[torchcodec \S+:\d+\] NVDEC not available or codec not supported; falling back to CPU decoding\.\n"
            # TODO: Remove once Python-side fallback log is removed.
            r"\[torchcodec \S+:\d+\] CUDA decoding fell back to CPU\.\n"
        )
        # fullmatch also validates that set_log_level("OFF") silenced logging
        assert re.fullmatch(expected, result.stderr), result.stderr

    def test_get_set_log_level(self):
        """Test that get_log_level reflects set_log_level changes."""
        from torchcodec._logging import get_log_level, set_log_level

        assert get_log_level() == "OFF"
        set_log_level("ALL")
        assert get_log_level() == "ALL"
        set_log_level("OFF")

    @needs_cuda
    def test_logging_disabled_by_default(self):
        """Test that no logs appear when logging is not explicitly enabled."""
        script = textwrap.dedent(
            """\
            from torchcodec.decoders import VideoDecoder, set_cuda_backend
            from test.utils import H265_VIDEO
            with set_cuda_backend("beta"):
                decoder = VideoDecoder(H265_VIDEO.path, device="cuda")
            # TODO: Remove this line once the Python-side
            # fallback log in _video_decoder.py is removed.
            _ = decoder.cpu_fallback
        """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert not result.stderr
