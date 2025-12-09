import io
import json
import os
import re
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest
import torch
from torchcodec.decoders import AudioDecoder, VideoDecoder

from torchcodec.encoders import AudioEncoder, VideoEncoder

from .utils import (
    assert_tensor_close_on_at_least,
    get_ffmpeg_major_version,
    get_ffmpeg_minor_version,
    in_fbcode,
    IN_GITHUB_CI,
    IS_WINDOWS,
    NASA_AUDIO_MP3,
    needs_ffmpeg_cli,
    psnr,
    SINE_MONO_S32,
    TEST_SRC_2_720P,
    TestContainerFile,
)

IS_WINDOWS_WITH_FFMPEG_LE_70 = IS_WINDOWS and (
    get_ffmpeg_major_version() < 7
    or (get_ffmpeg_major_version() == 7 and get_ffmpeg_minor_version() == 0)
)


@pytest.fixture
def with_ffmpeg_debug_logs():
    # Fixture that sets the ffmpeg logs to DEBUG mode
    previous_log_level = os.environ.get("TORCHCODEC_FFMPEG_LOG_LEVEL", "QUIET")
    os.environ["TORCHCODEC_FFMPEG_LOG_LEVEL"] = "DEBUG"
    yield
    os.environ["TORCHCODEC_FFMPEG_LOG_LEVEL"] = previous_log_level


def validate_frames_properties(*, actual: Path, expected: Path):
    # actual and expected are files containing encoded audio data.  We call
    # `ffprobe` on both, and assert that the frame properties match (pts,
    # duration, etc.)

    frames_actual, frames_expected = (
        json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-hide_banner",
                    "-select_streams",
                    "a:0",
                    "-show_frames",
                    "-of",
                    "json",
                    f"{f}",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )["frames"]
        for f in (actual, expected)
    )

    # frames_actual and frames_expected are both a list of dicts, each dict
    # corresponds to a frame and each key-value pair corresponds to a frame
    # property like pts, nb_samples, etc., similar to the AVFrame fields.
    assert isinstance(frames_actual, list)
    assert all(isinstance(d, dict) for d in frames_actual)

    assert len(frames_actual) > 3  # arbitrary sanity check
    assert len(frames_actual) == len(frames_expected)

    # non-exhaustive list of the props we want to test for:
    required_props = (
        "pts",
        "pts_time",
        "sample_fmt",
        "nb_samples",
        "channels",
        "duration",
        "duration_time",
    )

    for frame_index, (d_actual, d_expected) in enumerate(
        zip(frames_actual, frames_expected)
    ):
        if get_ffmpeg_major_version() >= 6:
            assert all(required_prop in d_expected for required_prop in required_props)

        for prop in d_expected:
            if prop == "pkt_pos":
                # pkt_pos is the position of the packet *in bytes* in its
                # stream. We don't always match FFmpeg exactly on this,
                # typically on compressed formats like mp3. It's probably
                # because we are not writing the exact same headers, or
                # something like this. In any case, this doesn't seem to be
                # critical.
                continue
            assert (
                d_actual[prop] == d_expected[prop]
            ), f"\nComparing: {actual}\nagainst reference: {expected},\nthe {prop} property is different at frame {frame_index}:"


class TestAudioEncoder:

    def decode(self, source) -> torch.Tensor:
        if isinstance(source, TestContainerFile):
            source = str(source.path)
        return AudioDecoder(source).get_all_samples()

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Expected samples to be a Tensor"):
            AudioEncoder(samples=123, sample_rate=32_000)
        with pytest.raises(ValueError, match="Expected 1D or 2D samples"):
            AudioEncoder(samples=torch.rand(3, 4, 5), sample_rate=32_000)
        with pytest.raises(ValueError, match="Expected float32 samples"):
            AudioEncoder(
                samples=torch.rand(10, 10, dtype=torch.float64), sample_rate=32_000
            )
        with pytest.raises(ValueError, match="sample_rate = 0 must be > 0"):
            AudioEncoder(samples=torch.rand(10, 10), sample_rate=0)

        encoder = AudioEncoder(samples=torch.rand(2, 100), sample_rate=32_000)

        bad_path = "/bad/path.mp3"
        with pytest.raises(
            RuntimeError,
            match=f"avio_open failed. The destination file is {bad_path}, make sure it's a valid path",
        ):
            encoder.to_file(dest=bad_path)

        bad_extension = "output.bad_extension"
        with pytest.raises(RuntimeError, match="check the desired extension"):
            encoder.to_file(dest=bad_extension)

        bad_format = "bad_format"
        with pytest.raises(
            RuntimeError,
            match=re.escape(f"Check the desired format? Got format={bad_format}"),
        ):
            encoder.to_tensor(format=bad_format)

    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    def test_bad_input_parametrized(self, method, tmp_path):
        if method == "to_file":
            valid_params = dict(dest=str(tmp_path / "output.mp3"))
        elif method == "to_tensor":
            valid_params = dict(format="mp3")
        elif method == "to_file_like":
            valid_params = dict(file_like=io.BytesIO(), format="mp3")
        else:
            raise ValueError(f"Unknown method: {method}")

        decoder = AudioEncoder(self.decode(NASA_AUDIO_MP3).data, sample_rate=10)
        avcodec_open2_failed_msg = "avcodec_open2 failed: Invalid argument"
        with pytest.raises(
            RuntimeError,
            match=(
                avcodec_open2_failed_msg
                if IS_WINDOWS_WITH_FFMPEG_LE_70
                else "invalid sample rate=10"
            ),
        ):
            getattr(decoder, method)(**valid_params)

        decoder = AudioEncoder(
            self.decode(NASA_AUDIO_MP3).data, sample_rate=NASA_AUDIO_MP3.sample_rate
        )
        with pytest.raises(
            RuntimeError,
            match=(
                avcodec_open2_failed_msg
                if IS_WINDOWS_WITH_FFMPEG_LE_70
                else "invalid sample rate=10"
            ),
        ):
            getattr(decoder, method)(sample_rate=10, **valid_params)
        with pytest.raises(
            RuntimeError,
            match=(
                avcodec_open2_failed_msg
                if IS_WINDOWS_WITH_FFMPEG_LE_70
                else "invalid sample rate=99999999"
            ),
        ):
            getattr(decoder, method)(sample_rate=99999999, **valid_params)
        with pytest.raises(RuntimeError, match="bit_rate=-1 must be >= 0"):
            getattr(decoder, method)(**valid_params, bit_rate=-1)

        bad_num_channels = 10
        decoder = AudioEncoder(torch.rand(bad_num_channels, 20), sample_rate=16_000)
        with pytest.raises(
            RuntimeError, match=f"Trying to encode {bad_num_channels} channels"
        ):
            getattr(decoder, method)(**valid_params)

        decoder = AudioEncoder(
            self.decode(NASA_AUDIO_MP3).data, sample_rate=NASA_AUDIO_MP3.sample_rate
        )
        for num_channels in (0, 3):
            match = (
                avcodec_open2_failed_msg
                if IS_WINDOWS_WITH_FFMPEG_LE_70
                else re.escape(
                    f"Desired number of channels ({num_channels}) is not supported"
                )
            )
            with pytest.raises(RuntimeError, match=match):
                getattr(decoder, method)(**valid_params, num_channels=num_channels)

    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    @pytest.mark.parametrize(
        "format",
        [
            pytest.param(
                "wav",
                marks=pytest.mark.skipif(
                    get_ffmpeg_major_version() == 4,
                    reason="Swresample with FFmpeg 4 doesn't work on wav files",
                ),
            ),
            "flac",
        ],
    )
    def test_round_trip(self, method, format, tmp_path):
        # Check that decode(encode(samples)) == samples on lossless formats

        asset = NASA_AUDIO_MP3
        source_samples = self.decode(asset).data

        encoder = AudioEncoder(source_samples, sample_rate=asset.sample_rate)

        if method == "to_file":
            encoded_path = str(tmp_path / f"output.{format}")
            encoded_source = encoded_path
            encoder.to_file(dest=encoded_path)
        elif method == "to_tensor":
            encoded_source = encoder.to_tensor(format=format)
            assert encoded_source.dtype == torch.uint8
            assert encoded_source.ndim == 1
        elif method == "to_file_like":
            file_like = io.BytesIO()
            encoder.to_file_like(file_like, format=format)
            encoded_source = file_like.getvalue()
        else:
            raise ValueError(f"Unknown method: {method}")

        rtol, atol = (0, 1e-4) if format == "wav" else (None, None)
        torch.testing.assert_close(
            self.decode(encoded_source).data, source_samples, rtol=rtol, atol=atol
        )

    @needs_ffmpeg_cli
    @pytest.mark.parametrize("asset", (NASA_AUDIO_MP3, SINE_MONO_S32))
    @pytest.mark.parametrize("bit_rate", (None, 0, 44_100, 999_999_999))
    @pytest.mark.parametrize("num_channels", (None, 1, 2))
    @pytest.mark.parametrize("sample_rate", (8_000, 32_000))
    @pytest.mark.parametrize("format", ("mp3", "wav", "flac"))
    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    def test_against_cli(
        self,
        asset,
        bit_rate,
        num_channels,
        sample_rate,
        format,
        method,
        tmp_path,
        capfd,
        with_ffmpeg_debug_logs,
    ):
        # Encodes samples with our encoder and with the FFmpeg CLI, and checks
        # that both decoded outputs are equal

        if get_ffmpeg_major_version() == 4 and format == "wav":
            pytest.skip("Swresample with FFmpeg 4 doesn't work on wav files")
        if IS_WINDOWS and get_ffmpeg_major_version() <= 5 and format == "mp3":
            # TODO: https://github.com/pytorch/torchcodec/issues/837
            pytest.skip("Encoding mp3 on Windows is weirdly buggy")

        encoded_by_ffmpeg = tmp_path / f"ffmpeg_output.{format}"
        subprocess.run(
            ["ffmpeg", "-i", str(asset.path)]
            + (["-b:a", f"{bit_rate}"] if bit_rate is not None else [])
            + (["-ac", f"{num_channels}"] if num_channels is not None else [])
            + ["-ar", f"{sample_rate}"]
            + [
                str(encoded_by_ffmpeg),
            ],
            capture_output=True,
            check=True,
        )

        encoder = AudioEncoder(self.decode(asset).data, sample_rate=asset.sample_rate)
        params = dict(
            bit_rate=bit_rate, num_channels=num_channels, sample_rate=sample_rate
        )
        if method == "to_file":
            encoded_by_us = tmp_path / f"output.{format}"
            encoder.to_file(dest=str(encoded_by_us), **params)
        elif method == "to_tensor":
            encoded_by_us = encoder.to_tensor(format=format, **params)
        elif method == "to_file_like":
            file_like = io.BytesIO()
            encoder.to_file_like(file_like, format=format, **params)
            encoded_by_us = file_like.getvalue()
        else:
            raise ValueError(f"Unknown method: {method}")

        captured = capfd.readouterr()
        if format == "wav":
            assert "Timestamps are unset in a packet" not in captured.err
        if format == "mp3":
            assert "Queue input is backward in time" not in captured.err
        if format in ("flac", "wav"):
            assert "Encoder did not produce proper pts" not in captured.err
        if format in ("flac", "mp3"):
            assert "Application provided invalid" not in captured.err

        assert_close = torch.testing.assert_close
        if sample_rate != asset.sample_rate:
            rtol, atol = 0, 1e-3
            if sys.platform == "darwin":
                assert_close = partial(assert_tensor_close_on_at_least, percentage=99)
        elif format == "wav":
            rtol, atol = 0, 1e-4
        elif format == "mp3" and asset is SINE_MONO_S32 and num_channels == 2:
            # Not sure why, this one needs slightly higher tol. With default
            # tolerances, the check fails on ~1% of the samples, so that's
            # probably fine. It might be that the FFmpeg CLI doesn't rely on
            # libswresample for converting channels?
            rtol, atol = 0, 1e-3
        else:
            rtol, atol = None, None

        if IS_WINDOWS_WITH_FFMPEG_LE_70 and format == "mp3":
            # We're getting a "Could not open input file" on Windows mp3 files when decoding.
            # TODO: https://github.com/pytorch/torchcodec/issues/837
            return

        samples_by_us = self.decode(encoded_by_us)
        samples_by_ffmpeg = self.decode(encoded_by_ffmpeg)

        assert_close(
            samples_by_us.data,
            samples_by_ffmpeg.data,
            rtol=rtol,
            atol=atol,
        )
        assert samples_by_us.pts_seconds == samples_by_ffmpeg.pts_seconds
        assert samples_by_us.duration_seconds == samples_by_ffmpeg.duration_seconds
        assert samples_by_us.sample_rate == samples_by_ffmpeg.sample_rate

        if method == "to_file":
            validate_frames_properties(actual=encoded_by_us, expected=encoded_by_ffmpeg)

    @pytest.mark.parametrize("asset", (NASA_AUDIO_MP3, SINE_MONO_S32))
    @pytest.mark.parametrize("bit_rate", (None, 0, 44_100, 999_999_999))
    @pytest.mark.parametrize("num_channels", (None, 1, 2))
    @pytest.mark.parametrize(
        "format",
        [
            # TODO: https://github.com/pytorch/torchcodec/issues/837
            pytest.param(
                "mp3",
                marks=pytest.mark.skipif(
                    IS_WINDOWS and get_ffmpeg_major_version() <= 5,
                    reason="Encoding mp3 on Windows is weirdly buggy",
                ),
            ),
            pytest.param(
                "wav",
                marks=pytest.mark.skipif(
                    get_ffmpeg_major_version() == 4,
                    reason="Swresample with FFmpeg 4 doesn't work on wav files",
                ),
            ),
            "flac",
        ],
    )
    @pytest.mark.parametrize("method", ("to_tensor", "to_file_like"))
    def test_against_to_file(
        self, asset, bit_rate, num_channels, format, tmp_path, method
    ):
        encoder = AudioEncoder(self.decode(asset).data, sample_rate=asset.sample_rate)

        params = dict(bit_rate=bit_rate, num_channels=num_channels)
        encoded_file = tmp_path / f"output.{format}"
        encoder.to_file(dest=encoded_file, **params)

        if method == "to_tensor":
            encoded_output = encoder.to_tensor(
                format=format, bit_rate=bit_rate, num_channels=num_channels
            )
        elif method == "to_file_like":
            file_like = io.BytesIO()
            encoder.to_file_like(
                file_like, format=format, bit_rate=bit_rate, num_channels=num_channels
            )
            encoded_output = file_like.getvalue()
        else:
            raise ValueError(f"Unknown method: {method}")

        if not (IS_WINDOWS_WITH_FFMPEG_LE_70 and format == "mp3"):
            # We're getting a "Could not open input file" on Windows mp3 files when decoding.
            # TODO: https://github.com/pytorch/torchcodec/issues/837
            torch.testing.assert_close(
                self.decode(encoded_file).data, self.decode(encoded_output).data
            )

    def test_encode_to_tensor_long_output(self):
        # Check that we support re-allocating the output tensor when the encoded
        # data is large.
        samples = torch.rand(1, int(1e7))
        encoded_tensor = AudioEncoder(samples, sample_rate=16_000).to_tensor(
            format="flac", bit_rate=44_000
        )

        # Note: this should be in sync with its C++ counterpart for the test to
        # be meaningful.
        INITIAL_TENSOR_SIZE = 10_000_000
        assert encoded_tensor.numel() > INITIAL_TENSOR_SIZE

        torch.testing.assert_close(self.decode(encoded_tensor).data, samples)

    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    def test_contiguity(self, method, tmp_path):
        # Ensure that 2 waveforms with the same values are encoded in the same
        # way, regardless of their memory layout. Here we encode 2 equal
        # waveforms, one is row-aligned while the other is column-aligned.

        num_samples = 10_000  # per channel
        contiguous_samples = torch.rand(2, num_samples).contiguous()
        assert contiguous_samples.stride() == (num_samples, 1)

        non_contiguous_samples = contiguous_samples.T.contiguous().T
        assert non_contiguous_samples.stride() == (1, 2)

        torch.testing.assert_close(
            contiguous_samples, non_contiguous_samples, rtol=0, atol=0
        )

        def encode_to_tensor(samples):
            params = dict(bit_rate=44_000)
            if method == "to_file":
                dest = str(tmp_path / "output.flac")
                AudioEncoder(samples, sample_rate=16_000).to_file(dest=dest, **params)
                with open(dest, "rb") as f:
                    return torch.frombuffer(f.read(), dtype=torch.uint8)
            elif method == "to_tensor":
                return AudioEncoder(samples, sample_rate=16_000).to_tensor(
                    format="flac", **params
                )
            elif method == "to_file_like":
                file_like = io.BytesIO()
                AudioEncoder(samples, sample_rate=16_000).to_file_like(
                    file_like, format="flac", **params
                )
                return torch.frombuffer(file_like.getvalue(), dtype=torch.uint8)
            else:
                raise ValueError(f"Unknown method: {method}")

        encoded_from_contiguous = encode_to_tensor(contiguous_samples)
        encoded_from_non_contiguous = encode_to_tensor(non_contiguous_samples)

        torch.testing.assert_close(
            encoded_from_contiguous, encoded_from_non_contiguous, rtol=0, atol=0
        )

    @pytest.mark.skip(
        reason="Flaky test, see https://github.com/pytorch/torchcodec/issues/724"
    )
    @pytest.mark.parametrize("num_channels_input", (1, 2))
    @pytest.mark.parametrize("num_channels_output", (1, 2, None))
    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    def test_num_channels(
        self, num_channels_input, num_channels_output, method, tmp_path
    ):
        # We just check that the num_channels parameter is respected.
        # Correctness is checked in other tests (like test_against_cli())

        sample_rate = 16_000
        source_samples = torch.rand(num_channels_input, 1_000)
        format = "flac"

        encoder = AudioEncoder(source_samples, sample_rate=sample_rate)
        params = dict(num_channels=num_channels_output)

        if method == "to_file":
            encoded_path = str(tmp_path / f"output.{format}")
            encoded_source = encoded_path
            encoder.to_file(dest=encoded_path, **params)
        elif method == "to_tensor":
            encoded_source = encoder.to_tensor(format=format, **params)
        elif method == "to_file_like":
            file_like = io.BytesIO()
            encoder.to_file_like(file_like, format=format, **params)
            encoded_source = file_like.getvalue()
        else:
            raise ValueError(f"Unknown method: {method}")

        if num_channels_output is None:
            num_channels_output = num_channels_input
        assert self.decode(encoded_source).data.shape[0] == num_channels_output

    def test_1d_samples(self):
        # smoke test making sure 1D samples are supported
        samples_1d, sample_rate = torch.rand(1000), 16_000
        samples_2d = samples_1d[None, :]

        torch.testing.assert_close(
            AudioEncoder(samples_1d, sample_rate=sample_rate).to_tensor("wav"),
            AudioEncoder(samples_2d, sample_rate=sample_rate).to_tensor("wav"),
        )

    def test_to_file_like_custom_file_object(self, tmp_path):
        class CustomFileObject:
            def __init__(self):
                self._file = io.BytesIO()

            def write(self, data):
                return self._file.write(data)

            def seek(self, offset, whence=0):
                return self._file.seek(offset, whence)

            def get_encoded_data(self):
                return self._file.getvalue()

        asset = NASA_AUDIO_MP3
        source_samples = self.decode(asset).data
        encoder = AudioEncoder(source_samples, sample_rate=asset.sample_rate)

        file_like = CustomFileObject()
        encoder.to_file_like(file_like, format="flac")

        decoded_samples = self.decode(file_like.get_encoded_data())

        torch.testing.assert_close(
            decoded_samples.data,
            source_samples,
            rtol=0,
            atol=1e-4,
        )

    def test_to_file_like_real_file(self, tmp_path):
        """Test to_file_like with a real file opened in binary write mode."""
        asset = NASA_AUDIO_MP3
        source_samples = self.decode(asset).data
        encoder = AudioEncoder(source_samples, sample_rate=asset.sample_rate)

        file_path = tmp_path / "test_file_like.wav"

        with open(file_path, "wb") as file_like:
            encoder.to_file_like(file_like, format="flac")

        decoded_samples = self.decode(str(file_path))
        torch.testing.assert_close(
            decoded_samples.data, source_samples, rtol=0, atol=1e-4
        )

    def test_to_file_like_bad_methods(self):
        asset = NASA_AUDIO_MP3
        source_samples = self.decode(asset).data
        encoder = AudioEncoder(source_samples, sample_rate=asset.sample_rate)

        class NoWriteMethod:
            def seek(self, offset, whence=0):
                return 0

        with pytest.raises(
            RuntimeError, match="File like object must implement a write method"
        ):
            encoder.to_file_like(NoWriteMethod(), format="wav")

        class NoSeekMethod:
            def write(self, data):
                return len(data)

        with pytest.raises(
            RuntimeError, match="File like object must implement a seek method"
        ):
            encoder.to_file_like(NoSeekMethod(), format="wav")


class TestVideoEncoder:
    def decode(self, source=None) -> torch.Tensor:
        return VideoDecoder(source).get_frames_in_range(start=0, stop=30).data

    # TODO: add average_fps field to TestVideo asset
    def decode_and_get_frame_rate(self, source=None):
        decoder = VideoDecoder(source)
        frames = decoder.get_frames_in_range(start=0, stop=30).data
        frame_rate = decoder.metadata.average_fps
        return frames, frame_rate

    def _get_video_metadata(self, file_path, fields):
        """Helper function to get video metadata from a file using ffprobe."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                f"stream={','.join(fields)}",
                "-of",
                "default=noprint_wrappers=1",
                str(file_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        metadata = {}
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key] = value
        assert all(field in metadata for field in fields)
        return metadata

    def _get_frames_info(self, file_path, fields):
        """Helper function to get frame info (pts, dts, etc.) using ffprobe."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                f"frame={','.join(fields)}",
                "-of",
                "json",
                str(file_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        frames = json.loads(result.stdout)["frames"]
        assert all(field in frame for field in fields for frame in frames)
        return frames

    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    def test_bad_input_parameterized(self, tmp_path, method):
        if method == "to_file":
            valid_params = dict(dest=str(tmp_path / "output.mp4"))
        elif method == "to_tensor":
            valid_params = dict(format="mp4")
        elif method == "to_file_like":
            valid_params = dict(file_like=io.BytesIO(), format="mp4")
        else:
            raise ValueError(f"Unknown method: {method}")

        with pytest.raises(
            ValueError, match="Expected uint8 frames, got frames.dtype = torch.float32"
        ):
            encoder = VideoEncoder(
                frames=torch.rand(5, 3, 64, 64),
                frame_rate=30,
            )
            getattr(encoder, method)(**valid_params)

        with pytest.raises(
            ValueError, match=r"Expected 4D frames, got frames.shape = torch.Size"
        ):
            encoder = VideoEncoder(
                frames=torch.zeros(10),
                frame_rate=30,
            )
            getattr(encoder, method)(**valid_params)

        with pytest.raises(
            RuntimeError, match=r"frame must have 3 channels \(R, G, B\), got 2"
        ):
            encoder = VideoEncoder(
                frames=torch.zeros((5, 2, 64, 64), dtype=torch.uint8),
                frame_rate=30,
            )
            getattr(encoder, method)(**valid_params)

        with pytest.raises(
            RuntimeError,
            match=r"Video codec invalid_codec_name not found.",
        ):
            encoder = VideoEncoder(
                frames=torch.zeros((5, 3, 64, 64), dtype=torch.uint8),
                frame_rate=30,
            )
            encoder.to_file(str(tmp_path / "output.mp4"), codec="invalid_codec_name")

        with pytest.raises(RuntimeError, match=r"crf=-10 is out of valid range"):
            encoder = VideoEncoder(
                frames=torch.zeros((5, 3, 64, 64), dtype=torch.uint8),
                frame_rate=30,
            )
            getattr(encoder, method)(**valid_params, crf=-10)

        with pytest.raises(
            RuntimeError,
            match=r"avcodec_open2 failed: Invalid argument",
        ):
            encoder.to_tensor(format="mp4", preset="fake_preset")

    @pytest.mark.parametrize("method", ["to_file", "to_tensor", "to_file_like"])
    @pytest.mark.parametrize("crf", [23, 23.5, -0.9])
    def test_crf_valid_values(self, method, crf, tmp_path):
        if method == "to_file":
            valid_params = {"dest": str(tmp_path / "test.mp4")}
        elif method == "to_tensor":
            valid_params = {"format": "mp4"}
        elif method == "to_file_like":
            valid_params = dict(file_like=io.BytesIO(), format="mp4")
        else:
            raise ValueError(f"Unknown method: {method}")

        encoder = VideoEncoder(
            frames=torch.zeros((5, 3, 64, 64), dtype=torch.uint8),
            frame_rate=30,
        )
        getattr(encoder, method)(**valid_params, crf=crf)

    def test_bad_input(self, tmp_path):
        encoder = VideoEncoder(
            frames=torch.zeros((5, 3, 64, 64), dtype=torch.uint8),
            frame_rate=30,
        )

        with pytest.raises(
            RuntimeError,
            match=r"Couldn't allocate AVFormatContext. The destination file is ./file.bad_extension, check the desired extension\?",
        ):
            encoder.to_file("./file.bad_extension")

        with pytest.raises(
            RuntimeError,
            match=r"avio_open failed. The destination file is ./bad/path.mp3, make sure it's a valid path\?",
        ):
            encoder.to_file("./bad/path.mp3")

        with pytest.raises(
            RuntimeError,
            match=r"Couldn't allocate AVFormatContext. Check the desired format\? Got format=bad_format",
        ):
            encoder.to_tensor(format="bad_format")

    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    @pytest.mark.parametrize(
        "device", ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda))
    )
    def test_pixel_format_errors(self, method, device, tmp_path):
        frames = torch.zeros((5, 3, 64, 64), dtype=torch.uint8).to(device)
        encoder = VideoEncoder(frames, frame_rate=30)

        if method == "to_file":
            valid_params = dict(dest=str(tmp_path / "output.mp4"))
        elif method == "to_tensor":
            valid_params = dict(format="mp4")
        elif method == "to_file_like":
            valid_params = dict(file_like=io.BytesIO(), format="mp4")

        if device == "cuda":
            with pytest.raises(
                RuntimeError,
                match="GPU Video encoding currently only supports the NV12 pixel format. Do not set pixel_format to use NV12",
            ):
                getattr(encoder, method)(**valid_params, pixel_format="yuv420p")
            return

        with pytest.raises(
            RuntimeError,
            match=r"Unknown pixel format: invalid_pix_fmt[\s\S]*Supported pixel formats.*yuv420p",
        ):
            getattr(encoder, method)(**valid_params, pixel_format="invalid_pix_fmt")

        with pytest.raises(
            RuntimeError,
            match=r"Specified pixel format rgb24 is not supported[\s\S]*Supported pixel formats.*yuv420p",
        ):
            getattr(encoder, method)(**valid_params, pixel_format="rgb24")

    @pytest.mark.parametrize(
        "extra_options,error",
        [
            ({"qp": -10}, "qp=-10 is out of valid range"),
            (
                {"qp": ""},
                "Option qp expects a numeric value but got",
            ),
            (
                {"direct-pred": "a"},
                "Option direct-pred expects a numeric value but got 'a'",
            ),
            ({"tune": "not_a_real_tune"}, "avcodec_open2 failed: Invalid argument"),
            (
                {"tune": 10},
                "avcodec_open2 failed: Invalid argument",
            ),
        ],
    )
    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    def test_extra_options_errors(self, method, tmp_path, extra_options, error):
        frames = torch.zeros((5, 3, 64, 64), dtype=torch.uint8)
        encoder = VideoEncoder(frames, frame_rate=30)

        if method == "to_file":
            valid_params = dict(dest=str(tmp_path / "output.mp4"))
        elif method == "to_tensor":
            valid_params = dict(format="mp4")
        elif method == "to_file_like":
            valid_params = dict(file_like=io.BytesIO(), format="mp4")
        else:
            raise ValueError(f"Unknown method: {method}")

        with pytest.raises(
            RuntimeError,
            match=error,
        ):
            getattr(encoder, method)(**valid_params, extra_options=extra_options)

    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    @pytest.mark.parametrize(
        "device",
        (
            "cpu",
            pytest.param(
                "cuda",
                marks=[
                    pytest.mark.needs_cuda,
                    pytest.mark.skipif(
                        in_fbcode(), reason="NVENC not available in fbcode"
                    ),
                ],
            ),
        ),
    )
    def test_contiguity(self, method, tmp_path, device):
        # Ensure that 2 sets of video frames with the same pixel values are encoded
        # in the same way, regardless of their memory layout. Here we encode 2 equal
        # frame tensors, one is contiguous while the other is non-contiguous.

        num_frames, channels, height, width = 5, 3, 256, 256
        contiguous_frames = (
            torch.randint(
                0, 256, size=(num_frames, channels, height, width), dtype=torch.uint8
            )
            .contiguous()
            .to(device)
        )
        assert contiguous_frames.is_contiguous()

        # Permute NCHW to NHWC, then update the memory layout, then permute back
        non_contiguous_frames = (
            contiguous_frames.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
        )
        assert non_contiguous_frames.stride() != contiguous_frames.stride()
        assert not non_contiguous_frames.is_contiguous()
        assert non_contiguous_frames.is_contiguous(memory_format=torch.channels_last)

        torch.testing.assert_close(
            contiguous_frames, non_contiguous_frames, rtol=0, atol=0
        )

        def encode_to_tensor(frames):
            common_params = dict(
                crf=0,
                pixel_format="yuv444p" if device == "cpu" else None,
                codec="h264_nvenc" if device != "cpu" else None,
            )
            if method == "to_file":
                dest = str(tmp_path / "output.mp4")
                VideoEncoder(frames, frame_rate=30).to_file(dest=dest, **common_params)
                with open(dest, "rb") as f:
                    return torch.frombuffer(f.read(), dtype=torch.uint8)
            elif method == "to_tensor":
                return VideoEncoder(frames, frame_rate=30).to_tensor(
                    format="mp4", **common_params
                )
            elif method == "to_file_like":
                file_like = io.BytesIO()
                VideoEncoder(frames, frame_rate=30).to_file_like(
                    file_like, format="mp4", **common_params
                )
                return torch.frombuffer(file_like.getvalue(), dtype=torch.uint8)
            else:
                raise ValueError(f"Unknown method: {method}")

        encoded_from_contiguous = encode_to_tensor(contiguous_frames)
        encoded_from_non_contiguous = encode_to_tensor(non_contiguous_frames)

        torch.testing.assert_close(
            encoded_from_contiguous, encoded_from_non_contiguous, rtol=0, atol=0
        )

    @pytest.mark.parametrize(
        "format",
        [
            "mov",
            "mp4",
            "mkv",
            pytest.param(
                "webm",
                marks=[
                    pytest.mark.slow,
                    pytest.mark.skipif(
                        get_ffmpeg_major_version() == 4
                        or (IS_WINDOWS and get_ffmpeg_major_version() in (6, 7)),
                        reason="Codec for webm is not available in this FFmpeg installation.",
                    ),
                ],
            ),
        ],
    )
    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    def test_round_trip(self, tmp_path, format, method):
        # Test that decode(encode(decode(frames))) == decode(frames)
        source_frames, frame_rate = self.decode_and_get_frame_rate(TEST_SRC_2_720P.path)

        encoder = VideoEncoder(frames=source_frames, frame_rate=frame_rate)

        if method == "to_file":
            encoded_path = str(tmp_path / f"encoder_output.{format}")
            encoder.to_file(dest=encoded_path, pixel_format="yuv444p", crf=0)
            round_trip_frames = self.decode(encoded_path)
        elif method == "to_tensor":
            encoded_tensor = encoder.to_tensor(
                format=format, pixel_format="yuv444p", crf=0
            )
            round_trip_frames = self.decode(encoded_tensor)
        elif method == "to_file_like":
            file_like = io.BytesIO()
            encoder.to_file_like(
                file_like=file_like, format=format, pixel_format="yuv444p", crf=0
            )
            round_trip_frames = self.decode(file_like.getvalue())
        else:
            raise ValueError(f"Unknown method: {method}")

        assert source_frames.shape == round_trip_frames.shape
        assert source_frames.dtype == round_trip_frames.dtype

        atol = 3 if format == "webm" else 2
        for s_frame, rt_frame in zip(source_frames, round_trip_frames):
            assert psnr(s_frame, rt_frame) > 30
            torch.testing.assert_close(s_frame, rt_frame, atol=atol, rtol=0)

    @pytest.mark.parametrize(
        "format",
        [
            "mov",
            "mp4",
            "avi",
            "mkv",
            "flv",
            "gif",
            pytest.param(
                "webm",
                marks=[
                    pytest.mark.slow,
                    pytest.mark.skipif(
                        get_ffmpeg_major_version() == 4
                        or (IS_WINDOWS and get_ffmpeg_major_version() in (6, 7)),
                        reason="Codec for webm is not available in this FFmpeg installation.",
                    ),
                ],
            ),
        ],
    )
    @pytest.mark.parametrize("method", ("to_tensor", "to_file_like"))
    def test_against_to_file(self, tmp_path, format, method):
        # Test that to_file, to_tensor, and to_file_like produce the same results
        source_frames, frame_rate = self.decode_and_get_frame_rate(TEST_SRC_2_720P.path)
        encoder = VideoEncoder(frames=source_frames, frame_rate=frame_rate)

        encoded_file = tmp_path / f"output.{format}"
        encoder.to_file(dest=encoded_file, crf=0)

        if method == "to_tensor":
            encoded_output = encoder.to_tensor(format=format, crf=0)
        else:  # to_file_like
            file_like = io.BytesIO()
            encoder.to_file_like(file_like=file_like, format=format, crf=0)
            encoded_output = file_like.getvalue()

        torch.testing.assert_close(
            self.decode(encoded_file),
            self.decode(encoded_output),
            atol=0,
            rtol=0,
        )

    @needs_ffmpeg_cli
    @pytest.mark.parametrize(
        "format",
        (
            "mov",
            "mp4",
            "avi",
            "mkv",
            "flv",
            pytest.param("webm", marks=pytest.mark.slow),
        ),
    )
    @pytest.mark.parametrize(
        "encode_params",
        [
            {"pixel_format": "yuv444p", "crf": 0, "preset": None},
            {"pixel_format": "yuv420p", "crf": 30, "preset": None},
            {"pixel_format": "yuv420p", "crf": None, "preset": "ultrafast"},
            {"pixel_format": "yuv420p", "crf": None, "preset": None},
        ],
    )
    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    @pytest.mark.parametrize("frame_rate", [30, 29.97])
    def test_video_encoder_against_ffmpeg_cli(
        self, tmp_path, format, encode_params, method, frame_rate
    ):
        ffmpeg_version = get_ffmpeg_major_version()
        if format == "webm" and (
            ffmpeg_version == 4 or (IS_WINDOWS and ffmpeg_version in (6, 7))
        ):
            pytest.skip("Codec for webm is not available in this FFmpeg installation.")

        pixel_format = encode_params["pixel_format"]
        crf = encode_params["crf"]
        preset = encode_params["preset"]

        if format in ("avi", "flv") and pixel_format == "yuv444p":
            pytest.skip(f"Default codec for {format} does not support {pixel_format}")

        source_frames = self.decode(TEST_SRC_2_720P.path)

        # Encode with FFmpeg CLI
        temp_raw_path = str(tmp_path / "temp_input.raw")
        with open(temp_raw_path, "wb") as f:
            f.write(source_frames.permute(0, 2, 3, 1).cpu().numpy().tobytes())

        ffmpeg_encoded_path = str(tmp_path / f"ffmpeg_output.{format}")
        # Some codecs (ex. MPEG4) do not support CRF or preset.
        # Flags not supported by the selected codec will be ignored.
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",  # Input format
            "-s",
            f"{source_frames.shape[3]}x{source_frames.shape[2]}",
            "-r",
            str(frame_rate),
            "-i",
            temp_raw_path,
        ]
        if pixel_format is not None:  # Output format
            ffmpeg_cmd.extend(["-pix_fmt", pixel_format])
        if preset is not None:
            ffmpeg_cmd.extend(["-preset", preset])
        if crf is not None:
            ffmpeg_cmd.extend(["-crf", str(crf)])
        # Output path must be last
        ffmpeg_cmd.append(ffmpeg_encoded_path)
        subprocess.run(ffmpeg_cmd, check=True)
        ffmpeg_frames = self.decode(ffmpeg_encoded_path).data

        # Encode with our video encoder
        encoder = VideoEncoder(frames=source_frames, frame_rate=frame_rate)
        encoder_output_path = str(tmp_path / f"encoder_output.{format}")

        if method == "to_file":
            encoder.to_file(
                dest=encoder_output_path,
                pixel_format=pixel_format,
                crf=crf,
                preset=preset,
            )
            encoder_frames = self.decode(encoder_output_path)
        elif method == "to_tensor":
            encoded_output = encoder.to_tensor(
                format=format,
                pixel_format=pixel_format,
                crf=crf,
                preset=preset,
            )
            encoder_frames = self.decode(encoded_output)
        elif method == "to_file_like":
            file_like = io.BytesIO()
            encoder.to_file_like(
                file_like=file_like,
                format=format,
                pixel_format=pixel_format,
                crf=crf,
                preset=preset,
            )
            encoder_frames = self.decode(file_like.getvalue())
        else:
            raise ValueError(f"Unknown method: {method}")

        assert ffmpeg_frames.shape[0] == encoder_frames.shape[0]

        # MPEG codec used for avi format does not accept CRF
        percentage = 94 if format == "avi" else 99

        # Check that PSNR between both encoded versions is high
        for ff_frame, enc_frame in zip(ffmpeg_frames, encoder_frames):
            res = psnr(ff_frame, enc_frame)
            assert res > 30
            assert_tensor_close_on_at_least(
                ff_frame, enc_frame, percentage=percentage, atol=2
            )

        # Only compare video metadata on ffmpeg versions >= 6, as older versions
        # are often missing metadata
        if ffmpeg_version >= 6 and method == "to_file":
            fields = [
                "duration",
                "duration_ts",
                "r_frame_rate",
                "time_base",
                "nb_frames",
            ]
            ffmpeg_metadata = self._get_video_metadata(
                ffmpeg_encoded_path,
                fields=fields,
            )
            encoder_metadata = self._get_video_metadata(
                encoder_output_path,
                fields=fields,
            )
            assert ffmpeg_metadata == encoder_metadata

            # Check that frame timestamps and duration are the same
            fields = ("pts", "pts_time")
            if format != "flv":
                fields += ("duration", "duration_time")
            ffmpeg_frames_info = self._get_frames_info(
                ffmpeg_encoded_path, fields=fields
            )
            encoder_frames_info = self._get_frames_info(
                encoder_output_path, fields=fields
            )
            assert ffmpeg_frames_info == encoder_frames_info

    def test_to_file_like_custom_file_object(self):
        """Test to_file_like with a custom file-like object that implements write and seek."""

        class CustomFileObject:
            def __init__(self):
                self._file = io.BytesIO()

            def write(self, data):
                return self._file.write(data)

            def seek(self, offset, whence=0):
                return self._file.seek(offset, whence)

            def get_encoded_data(self):
                return self._file.getvalue()

        source_frames, frame_rate = self.decode_and_get_frame_rate(TEST_SRC_2_720P.path)
        encoder = VideoEncoder(frames=source_frames, frame_rate=frame_rate)

        file_like = CustomFileObject()
        encoder.to_file_like(file_like, format="mp4", pixel_format="yuv444p", crf=0)
        decoded_frames = self.decode(file_like.get_encoded_data())

        torch.testing.assert_close(
            decoded_frames,
            source_frames,
            atol=2,
            rtol=0,
        )

    def test_to_file_like_real_file(self, tmp_path):
        """Test to_file_like with a real file opened in binary write mode."""
        source_frames, frame_rate = self.decode_and_get_frame_rate(TEST_SRC_2_720P.path)
        encoder = VideoEncoder(frames=source_frames, frame_rate=frame_rate)

        file_path = tmp_path / "test_file_like.mp4"

        with open(file_path, "wb") as file_like:
            encoder.to_file_like(file_like, format="mp4", pixel_format="yuv444p", crf=0)
        decoded_frames = self.decode(str(file_path))

        torch.testing.assert_close(
            decoded_frames,
            source_frames,
            atol=2,
            rtol=0,
        )

    def test_to_file_like_bad_methods(self):
        source_frames, frame_rate = self.decode_and_get_frame_rate(TEST_SRC_2_720P.path)
        encoder = VideoEncoder(frames=source_frames, frame_rate=frame_rate)

        class NoWriteMethod:
            def seek(self, offset, whence=0):
                return 0

        with pytest.raises(
            RuntimeError, match="File like object must implement a write method"
        ):
            encoder.to_file_like(NoWriteMethod(), format="mp4")

        class NoSeekMethod:
            def write(self, data):
                return len(data)

        with pytest.raises(
            RuntimeError, match="File like object must implement a seek method"
        ):
            encoder.to_file_like(NoSeekMethod(), format="mp4")

    @needs_ffmpeg_cli
    @pytest.mark.parametrize(
        "format,codec_spec",
        [
            ("mp4", "h264"),
            ("mp4", "hevc"),
            ("mkv", "av1"),
            ("avi", "mpeg4"),
            pytest.param(
                "webm",
                "vp9",
                marks=pytest.mark.skipif(
                    IS_WINDOWS, reason="vp9 codec not available on Windows"
                ),
            ),
        ],
    )
    def test_codec_parameter_utilized(self, tmp_path, format, codec_spec):
        # Test the codec parameter is utilized by using ffprobe to check the encoded file's codec spec
        frames = torch.zeros((10, 3, 64, 64), dtype=torch.uint8)
        dest = str(tmp_path / f"output.{format}")

        VideoEncoder(frames=frames, frame_rate=30).to_file(dest=dest, codec=codec_spec)
        actual_codec_spec = self._get_video_metadata(dest, fields=["codec_name"])[
            "codec_name"
        ]
        assert actual_codec_spec == codec_spec

    @needs_ffmpeg_cli
    @pytest.mark.parametrize(
        "codec_spec,codec_impl",
        [
            ("h264", "libx264"),
            ("av1", "libaom-av1"),
            pytest.param(
                "vp9",
                "libvpx-vp9",
                marks=pytest.mark.skipif(
                    IS_WINDOWS, reason="vp9 codec not available on Windows"
                ),
            ),
        ],
    )
    def test_codec_spec_vs_impl_equivalence(self, tmp_path, codec_spec, codec_impl):
        # Test that using codec spec gives the same result as using default codec implementation
        # We cannot directly check codec impl used, so we assert frame equality
        frames = torch.randint(0, 256, (10, 3, 64, 64), dtype=torch.uint8)

        spec_output = str(tmp_path / "spec_output.mp4")
        VideoEncoder(frames=frames, frame_rate=30).to_file(
            dest=spec_output, codec=codec_spec
        )

        impl_output = str(tmp_path / "impl_output.mp4")
        VideoEncoder(frames=frames, frame_rate=30).to_file(
            dest=impl_output, codec=codec_impl
        )

        assert (
            self._get_video_metadata(spec_output, fields=["codec_name"])["codec_name"]
            == codec_spec
        )
        assert (
            self._get_video_metadata(impl_output, fields=["codec_name"])["codec_name"]
            == codec_spec
        )

        frames_spec = self.decode(spec_output)
        frames_impl = self.decode(impl_output)
        torch.testing.assert_close(frames_spec, frames_impl, rtol=0, atol=0)

    @needs_ffmpeg_cli
    @pytest.mark.parametrize(
        "profile,colorspace,color_range",
        [
            ("baseline", "bt709", "tv"),
            ("main", "bt470bg", "pc"),
            ("high", "fcc", "pc"),
        ],
    )
    def test_extra_options_utilized(self, tmp_path, profile, colorspace, color_range):
        # Test setting profile, colorspace, and color_range via extra_options is utilized
        source_frames = torch.zeros((5, 3, 64, 64), dtype=torch.uint8)
        encoder = VideoEncoder(frames=source_frames, frame_rate=30)

        output_path = str(tmp_path / "output.mp4")
        encoder.to_file(
            dest=output_path,
            extra_options={
                "profile": profile,
                "colorspace": colorspace,
                "color_range": color_range,
            },
        )
        metadata = self._get_video_metadata(
            output_path,
            fields=["profile", "color_space", "color_range"],
        )
        # Validate profile (case-insensitive, baseline is reported as "Constrained Baseline")
        expected_profile = "constrained baseline" if profile == "baseline" else profile
        assert metadata["profile"].lower() == expected_profile
        assert metadata["color_space"] == colorspace
        assert metadata["color_range"] == color_range

    @needs_ffmpeg_cli
    @pytest.mark.needs_cuda
    # TODO-VideoEncoder: Auto-select codec for GPU encoding
    @pytest.mark.parametrize(
        "format_codec",
        [
            ("mov", "h264_nvenc"),
            ("mp4", "hevc_nvenc"),
            ("avi", "h264_nvenc"),
            pytest.param(
                ("mkv", "av1_nvenc"),
                marks=pytest.mark.skipif(
                    IN_GITHUB_CI, reason="av1_nvenc is not supported on CI"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    # TODO-VideoEncoder: Enable additional pixel formats ("yuv420p", "yuv444p")
    def test_nvenc_against_ffmpeg_cli(self, tmp_path, format_codec, method):
        # Encode with FFmpeg CLI using nvenc codecs
        format, codec = format_codec
        device = "cuda"
        qp = 1  # Lossless (qp=0) is not supported on av1_nvenc, so we use 1
        source_frames = self.decode(TEST_SRC_2_720P.path).data.to(device)

        temp_raw_path = str(tmp_path / "temp_input.raw")
        with open(temp_raw_path, "wb") as f:
            f.write(source_frames.permute(0, 2, 3, 1).cpu().numpy().tobytes())

        ffmpeg_encoded_path = str(tmp_path / f"ffmpeg_nvenc_output.{format}")
        frame_rate = 30

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",  # Input format
            "-s",
            f"{source_frames.shape[3]}x{source_frames.shape[2]}",
            "-r",
            str(frame_rate),
            "-i",
            temp_raw_path,
            "-c:v",
            codec,  # Use specified NVENC hardware encoder
        ]

        ffmpeg_cmd.extend(["-pix_fmt", "nv12"])  # Output format is always NV12
        if codec == "av1_nvenc":
            ffmpeg_cmd.extend(["-rc", "constqp"])  # Set rate control mode for AV1
        ffmpeg_cmd.extend(["-qp", str(qp)])  # Use lossless qp for other codecs
        ffmpeg_cmd.extend([ffmpeg_encoded_path])
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        encoder = VideoEncoder(frames=source_frames, frame_rate=frame_rate)

        encoder_extra_options = {"qp": qp}
        if codec == "av1_nvenc":
            encoder_extra_options["rc"] = 0  # constqp mode
        if method == "to_file":
            encoder_output_path = str(tmp_path / f"nvenc_output.{format}")
            encoder.to_file(
                dest=encoder_output_path,
                codec=codec,
                extra_options=encoder_extra_options,
            )
            encoder_output = encoder_output_path
        elif method == "to_tensor":
            encoder_output = encoder.to_tensor(
                format=format,
                codec=codec,
                extra_options=encoder_extra_options,
            )
        elif method == "to_file_like":
            file_like = io.BytesIO()
            encoder.to_file_like(
                file_like=file_like,
                format=format,
                codec=codec,
                extra_options=encoder_extra_options,
            )
            encoder_output = file_like.getvalue()
        else:
            raise ValueError(f"Unknown method: {method}")

        ffmpeg_frames = self.decode(ffmpeg_encoded_path).data
        encoder_frames = self.decode(encoder_output).data

        assert ffmpeg_frames.shape[0] == encoder_frames.shape[0]
        for ff_frame, enc_frame in zip(ffmpeg_frames, encoder_frames):
            assert psnr(ff_frame, enc_frame) > 25
            assert_tensor_close_on_at_least(ff_frame, enc_frame, percentage=96, atol=2)

        if method == "to_file":
            ffmpeg_metadata = self._get_video_metadata(ffmpeg_encoded_path, ["pix_fmt"])
            encoder_metadata = self._get_video_metadata(encoder_output, ["pix_fmt"])
            # pix_fmt nv12 is stored as yuv420p in metadata
            assert encoder_metadata["pix_fmt"] == "yuv420p"
            assert ffmpeg_metadata["pix_fmt"] == "yuv420p"
