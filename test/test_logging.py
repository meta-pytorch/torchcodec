# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
import textwrap

import pytest
from torchcodec import ffmpeg_major_version
from torchcodec._logging import get_log_level, set_log_level

from .utils import in_fbcode, needs_cuda


@pytest.fixture
def with_restore_log_level():
    current_log_level = get_log_level()
    yield
    set_log_level(current_log_level)


class TestLogging:
    @needs_cuda
    @pytest.mark.parametrize("log_level", ("ALL", "OFF", "default"))
    def test_cpp_logger(self, log_level):
        script = textwrap.dedent(
            f"""\
            from torchcodec._logging import set_log_level
            # the if below is a hacky way of validating that logging is off by default
            if {log_level != "default"}:
                set_log_level("{log_level}")
            from torchcodec.decoders import VideoDecoder
            from test.utils import H265_VIDEO
            decoder = VideoDecoder(H265_VIDEO.path, device="cuda")
        """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        expected = "Video stream not supported by NVDEC; falling back to CPU decoding"
        if log_level == "ALL":
            assert expected in result.stderr
        else:
            assert log_level in ("OFF", "default")
            assert expected not in result.stderr

    @pytest.mark.parametrize("log_level", ("ALL", "OFF"))
    def test_python_logger(self, log_level):
        script = textwrap.dedent(
            """\
            import torchcodec
        """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        if log_level == "ALL":
            # We log something when we fail to load ffmpeg and we first try
            # ffmpeg 8, then 7, etc. If we're on ffmpeg 8, we won't log
            # anything, since the loading will work from the first time, hence
            # why we skip this check on ffmpeg 8.
            # Similarly on fbcode, we never log anything here because we don't
            # hit the same code path
            if ffmpeg_major_version != 8 and not in_fbcode():
                assert "failed to load" in result.stderr
        else:
            assert log_level == "OFF"
            assert not result.stderr

    def test_get_set_log_level(self, with_restore_log_level):
        """Test that get_log_level reflects set_log_level changes."""

        assert get_log_level() == "OFF"
        set_log_level("ALL")
        assert get_log_level() == "ALL"
        set_log_level("OFF")
        assert get_log_level() == "OFF"
