# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
import textwrap

from .utils import needs_cuda


class TestLogging:
    @needs_cuda
    def test_cpp_logging_cuda_fallback(self):
        """Test that C++ TC_LOG fires when BetaCuda falls back to CPU."""
        script = textwrap.dedent(
            """\
            from torchcodec.decoders import VideoDecoder, set_cuda_backend
            from test.utils import H265_VIDEO
            with set_cuda_backend("beta"):
                decoder = VideoDecoder(H265_VIDEO.path, device="cuda")
        """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[torchcodec" in result.stderr
        assert "falling back to CPU" in result.stderr

    @needs_cuda
    def test_python_logging_cuda_fallback(self):
        """Test that Python logging fires when CPU fallback is detected."""
        script = textwrap.dedent(
            """\
            from torchcodec.decoders import VideoDecoder, set_cuda_backend
            from test.utils import H265_VIDEO
            with set_cuda_backend("beta"):
                decoder = VideoDecoder(H265_VIDEO.path, device="cuda")
            _ = decoder.cpu_fallback
        """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "CUDA decoding fell back to CPU" in result.stderr
