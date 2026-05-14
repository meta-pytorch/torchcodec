import io
import os
import platform
import re
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest
import torch
from torchcodec import ffmpeg_major_version
from torchcodec.decoders import AudioDecoder, VideoDecoder

from torchcodec.encoders import AudioEncoder, VideoEncoder
from torchcodec.encoders._multi_stream_encoder import StreamingEncoder

from .utils import (
    assert_tensor_close_on_at_least,
    call_ffprobe,
    get_ffmpeg_minor_version,
    in_fbcode,
    IN_GITHUB_CI,
    IS_WINDOWS,
    NASA_AUDIO_MP3,
    NASA_AUDIO_MP3_44100,
    NASA_VIDEO,
    needs_ffmpeg_cli,
    psnr,
    SINE_MONO_S32,
    TEST_SRC_2_720P,
    TestContainerFile,
)

IS_WINDOWS_WITH_FFMPEG_LE_70 = IS_WINDOWS and (
    ffmpeg_major_version < 7
    or (ffmpeg_major_version == 7 and get_ffmpeg_minor_version() == 0)
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
    show_entries = "frame=" + ",".join(required_props)

    frames_actual, frames_expected = (
        call_ffprobe(
            [
                "-select_streams",
                "a:0",
                "-show_frames",
                "-show_entries",
                show_entries,
                f"{f}",
            ]
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

    for frame_index, (d_actual, d_expected) in enumerate(
        zip(frames_actual, frames_expected)
    ):
        if ffmpeg_major_version >= 6:
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
                    ffmpeg_major_version == 4,
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
    @pytest.mark.parametrize(
        "format",
        [
            # TODO: https://github.com/pytorch/torchcodec/issues/837
            pytest.param(
                "mp3",
                marks=pytest.mark.skipif(
                    IS_WINDOWS and ffmpeg_major_version <= 5,
                    reason="Encoding mp3 on Windows is weirdly buggy",
                ),
            ),
            pytest.param(
                "wav",
                marks=pytest.mark.skipif(
                    ffmpeg_major_version == 4,
                    reason="Swresample with FFmpeg 4 doesn't work on wav files",
                ),
            ),
            "flac",
        ],
    )
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
            if platform.machine().lower() == "aarch64":
                rtol, atol = 0, 1e-2
            else:
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
                    IS_WINDOWS and ffmpeg_major_version <= 5,
                    reason="Encoding mp3 on Windows is weirdly buggy",
                ),
            ),
            pytest.param(
                "wav",
                marks=pytest.mark.skipif(
                    ffmpeg_major_version == 4,
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
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
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
        parsed = call_ffprobe(
            [
                "-select_streams",
                "v:0",
                "-show_entries",
                f"frame={','.join(fields)}",
                str(file_path),
            ]
        )
        frames = parsed["frames"]
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
                match="Video encoding on GPU currently only supports the nv12 pixel format. Do not set pixel_format to use nv12 by default.",
            ):
                getattr(encoder, method)(**valid_params, pixel_format="yuv444p")
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
                    pytest.mark.skipif(
                        ffmpeg_major_version == 4,
                        reason="CUDA + FFmpeg 4 test is flaky",
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
                        ffmpeg_major_version == 4
                        or (IS_WINDOWS and ffmpeg_major_version >= 6),
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
                        ffmpeg_major_version == 4
                        or (IS_WINDOWS and ffmpeg_major_version >= 6),
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
            pytest.param(
                "webm",
                marks=[
                    pytest.mark.slow,
                    pytest.mark.skipif(
                        ffmpeg_major_version == 4
                        or (IS_WINDOWS and ffmpeg_major_version >= 6),
                        reason="Codec for webm is not available in this FFmpeg installation.",
                    ),
                ],
            ),
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
        if ffmpeg_major_version >= 6 and method == "to_file":
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
    @pytest.mark.parametrize("method", ("to_file", "to_tensor", "to_file_like"))
    @pytest.mark.parametrize(
        ("format", "codec"),
        [
            ("mov", None),  # will default to h264_nvenc
            ("mov", "h264_nvenc"),
            ("avi", "h264_nvenc"),
            ("mp4", "hevc_nvenc"),  # use non-default codec
            pytest.param(
                "mkv",
                "av1_nvenc",
                marks=[
                    pytest.mark.skipif(
                        IN_GITHUB_CI, reason="av1_nvenc is not supported on CI"
                    ),
                    pytest.mark.skipif(
                        ffmpeg_major_version == 4,
                        reason="av1_nvenc is not supported on FFmpeg 4",
                    ),
                ],
            ),
        ],
    )
    # We test the color space and color range parameters in this test, because
    # we are required to define matrices specific to these specs when using NPP, see note:
    # [RGB -> YUV Color Conversion, limited color range]
    # BT.601, BT.709, BT.2020
    @pytest.mark.parametrize("color_space", ("bt470bg", "bt709", "bt2020nc", None))
    # Full/PC range, Limited/TV range
    @pytest.mark.parametrize("color_range", ("pc", "tv", None))
    def test_nvenc_against_ffmpeg_cli(
        self, tmp_path, method, format, codec, color_space, color_range
    ):
        # TODO-VideoEncoder: (P2) Investigate why FFmpeg 4 and 6 fail with non-default color space and range.
        # See https://github.com/meta-pytorch/torchcodec/issues/1140
        if ffmpeg_major_version in (4, 6) and not (
            color_space == "bt470bg" and color_range == "tv"
        ):
            pytest.skip(
                "Non-default color space and range have lower accuracy on FFmpeg 4 and 6"
            )
        # Encode with FFmpeg CLI using nvenc codecs
        device = "cuda"
        qp = 1  # Use near lossless encoding to reduce noise and support av1_nvenc
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
        ]
        # CLI requires explicit codec for nvenc
        # VideoEncoder will default to h264_nvenc since the frames are on GPU.
        ffmpeg_cmd.extend(["-c:v", codec if codec is not None else "h264_nvenc"])
        ffmpeg_cmd.extend(["-pix_fmt", "nv12"])  # Output format is always NV12
        ffmpeg_cmd.extend(["-qp", str(qp)])
        if color_space:
            ffmpeg_cmd.extend(["-colorspace", color_space])
        if color_range:
            ffmpeg_cmd.extend(["-color_range", color_range])
        ffmpeg_cmd.extend([ffmpeg_encoded_path])
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

        encoder = VideoEncoder(frames=source_frames, frame_rate=frame_rate)
        encoder_extra_options = {"qp": qp}
        if color_space:
            encoder_extra_options["colorspace"] = color_space
        if color_range:
            encoder_extra_options["color_range"] = color_range
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
            metadata_fields = ["pix_fmt", "color_range", "color_space"]
            ffmpeg_metadata = self._get_video_metadata(
                ffmpeg_encoded_path, metadata_fields
            )
            encoder_metadata = self._get_video_metadata(encoder_output, metadata_fields)
            # pix_fmt nv12 is stored as yuv420p in metadata, unless full range (pc)is used
            # In that case, h264 and hevc NVENC codecs will use yuvj420p automatically.
            if color_range == "pc" and codec != "av1_nvenc":
                expected_pix_fmt = "yuvj420p"
            else:
                # av1_nvenc does not utilize the yuvj420p pixel format
                expected_pix_fmt = "yuv420p"
            assert (
                encoder_metadata["pix_fmt"]
                == ffmpeg_metadata["pix_fmt"]
                == expected_pix_fmt
            )

            assert encoder_metadata["color_range"] == ffmpeg_metadata["color_range"]
            assert encoder_metadata["color_space"] == ffmpeg_metadata["color_space"]
            # Default values vary by codec, so we only assert when
            # color_range and color_space are not None.
            if color_range is not None:
                # FFmpeg and torchcodec encode color_range as 'unknown' for mov and avi
                # when color_range='tv' and color_space=None on FFmpeg 7/8.
                # Since this failure is rare, I suspect its a bug related to these
                # older container formats on newer FFmpeg versions.
                if not (
                    ffmpeg_major_version in (7, 8)
                    and color_range == "tv"
                    and color_space is None
                    and format in ("mov", "avi")
                ):
                    assert color_range == encoder_metadata["color_range"]
            if color_space is not None:
                assert color_space == encoder_metadata["color_space"]

    @pytest.mark.skipif(
        ffmpeg_major_version == 4,
        reason="On FFmpeg 4  hitting a truncated packet results in AVERROR_INVALIDDATA, which torchcodec does not handle.",
    )
    @pytest.mark.parametrize("format", ["mp4", "mov"])
    @pytest.mark.parametrize(
        "extra_options",
        [
            # frag_keyframe with empty_moov (new fragment every keyframe)
            {"movflags": "+frag_keyframe+empty_moov"},
            # frag_duration creates fragments based on duration (in microseconds)
            {"movflags": "+empty_moov", "frag_duration": "1000000"},
        ],
    )
    def test_fragmented_mp4(
        self,
        tmp_path,
        extra_options,
        format,
    ):
        # Test that VideoEncoder can write fragmented files using movflags.
        # Fragmented files store metadata interleaved with data rather than
        # all at the end, making them decodable even if writing is interrupted.
        source_frames, frame_rate = self.decode_and_get_frame_rate(TEST_SRC_2_720P.path)
        encoder = VideoEncoder(frames=source_frames, frame_rate=frame_rate)
        encoded_path = str(tmp_path / f"fragmented_output.{format}")
        encoder.to_file(dest=encoded_path, extra_options=extra_options)

        # Decode the file to get reference frames
        reference_decoder = VideoDecoder(encoded_path)
        reference_frames = [reference_decoder.get_frame_at(i) for i in range(10)]

        # Truncate the file to simulate interrupted write
        with open(encoded_path, "rb") as f:
            full_content = f.read()
        truncated_size = int(len(full_content) * 0.5)
        with open(encoded_path, "wb") as f:
            f.write(full_content[:truncated_size])

        # Decode the truncated file and verify first 10 frames match reference
        truncated_decoder = VideoDecoder(encoded_path)
        assert len(truncated_decoder) >= 10
        for i in range(10):
            truncated_frame = truncated_decoder.get_frame_at(i)
            torch.testing.assert_close(
                truncated_frame.data, reference_frames[i].data, atol=0, rtol=0
            )


class TestStreamingEncoder:
    cpu_and_oss_cuda = (
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.needs_cuda,
                pytest.mark.skipif(in_fbcode(), reason="NVENC not available in fbcode"),
            ],
        ),
    )

    @staticmethod
    def _create_encoder(method, tmp_path, format):
        encoder = StreamingEncoder()
        if method == "to_file":
            encoder_output = tmp_path / f"test.{format}"
            open_kwargs = dict(dest=encoder_output)
        elif method == "to_file_like":
            encoder_output = io.BytesIO()
            open_kwargs = dict(dest=encoder_output, format=format)
        else:
            raise ValueError(f"Unknown method: {method}")
        return encoder, encoder_output, open_kwargs

    @staticmethod
    def _get_decoder_source(encoder_output):
        if isinstance(encoder_output, io.BytesIO):
            return encoder_output.getvalue()
        return str(encoder_output)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_double_close(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.close()
        enc.close()  # double close is a no-op

    @pytest.mark.parametrize("format", ["mp4", "mov", "mkv"])
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("device", cpu_and_oss_cuda)
    def test_add_video_and_encode_frames(self, tmp_path, format, method, device):
        source_decoder = VideoDecoder(str(TEST_SRC_2_720P.path))
        source_frames = source_decoder.get_frames_in_range(start=0, stop=10).data.to(
            device
        )
        frame_rate = source_decoder.metadata.average_fps
        percentage, atol = (96, 2) if device == "cuda" else (99, 2)

        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        add_video_kwargs = {
            "height": source_frames.shape[2],
            "width": source_frames.shape[3],
            "frame_rate": frame_rate,
            "device": device,
        }
        if device == "cpu":
            add_video_kwargs["pixel_format"] = "yuv444p"
            add_video_kwargs["crf"] = 0
        else:
            add_video_kwargs["extra_options"] = {"qp": "1"}
        video = enc.add_video(**add_video_kwargs)
        enc.open(**open_kwargs)
        video.write(source_frames[:5])
        video.write(source_frames[5:])
        enc.close()

        decoded_frames = (
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=10)
            .data
        )
        assert_tensor_close_on_at_least(
            decoded_frames, source_frames.cpu(), percentage=percentage, atol=atol
        )

    def test_open_invalid_path(self):
        enc = StreamingEncoder()
        enc.add_video(height=64, width=64, frame_rate=30.0)
        with pytest.raises(RuntimeError, match="make sure it's a valid path"):
            enc.open("/nonexistent/dir/test.mp4")

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_open_invalid_format(self, tmp_path, method):
        enc = StreamingEncoder()
        enc.add_video(height=64, width=64, frame_rate=30.0)
        if method == "to_file":
            with pytest.raises(RuntimeError, match="check the desired extension"):
                enc.open(tmp_path / "test.bad_extension")
        elif method == "to_file_like":
            with pytest.raises(
                RuntimeError,
                match=r"Check the desired format\? Got format=bad_extension",
            ):
                enc.open(io.BytesIO(), format="bad_extension")

    @pytest.mark.parametrize("format", ["mp4", "mov"])
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("device", cpu_and_oss_cuda)
    def test_fragmented_mp4(self, format, tmp_path, method, device):
        source_decoder = VideoDecoder(str(TEST_SRC_2_720P.path))
        source_frames = source_decoder.get_frames_in_range(start=0, stop=10).data.to(
            device
        )
        frame_rate = source_decoder.metadata.average_fps
        percentage, atol = (96, 2) if device == "cuda" else (99, 2)

        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        # In addition to the fragmentation flag, "flush_packets" and "threads"
        # are necessary to decode frames before close().
        # See frag flags: https://ffmpeg.org/ffmpeg-formats.html#Fragmentation
        # TODO MultiStreamEncoder: Get a better understanding of which options
        # are necessary for reading fragmented mp4s
        extra_options = {
            "movflags": "+frag_every_frame+empty_moov",
            "flush_packets": "1",
            "threads": "1",
        }
        if device == "cuda":
            extra_options.update({"qp": "1", "delay": "0"})
            pixel_format, crf = None, None
        else:
            extra_options["tune"] = "zerolatency"
            pixel_format, crf = "yuv444p", 0
        video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=frame_rate,
            device=device,
            pixel_format=pixel_format,
            crf=crf,
            extra_options=extra_options,
        )
        enc.open(**open_kwargs)
        # Here, we decode the available fragmented mp4 frames before calling close()
        for batch in [source_frames[:5], source_frames[5:]]:
            video.write(batch)
            mid_decoder = VideoDecoder(self._get_decoder_source(encoder_output))
            num_available = len(mid_decoder)
            assert num_available > 0
            assert_tensor_close_on_at_least(
                mid_decoder.get_frames_in_range(start=0, stop=num_available).data,
                source_frames[:num_available].cpu(),
                percentage=percentage,
                atol=atol,
            )

        enc.close()
        # After close, all frames must be decodable
        assert_tensor_close_on_at_least(
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=10)
            .data,
            source_frames.cpu(),
            percentage=percentage,
            atol=atol,
        )

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("device", cpu_and_oss_cuda)
    def test_write_frames_mismatched_dimensions_errors(self, tmp_path, method, device):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=256, width=256, frame_rate=30.0, device=device)
        enc.open(**open_kwargs)
        # write with wrong size errors
        frames_128 = torch.randint(
            0, 256, (2, 3, 128, 128), dtype=torch.uint8, device=device
        )
        with pytest.raises(RuntimeError, match="same dimensions"):
            video.write(frames_128)
        # write with different size than first also errors
        frames_256 = torch.randint(
            0, 256, (2, 3, 256, 256), dtype=torch.uint8, device=device
        )
        frames_512 = torch.randint(
            0, 256, (2, 3, 512, 512), dtype=torch.uint8, device=device
        )
        video.write(frames_256)
        with pytest.raises(RuntimeError, match="same dimensions"):
            video.write(frames_512)

    @pytest.mark.needs_cuda
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_frames_different_devices_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=30.0)
        enc.open(**open_kwargs)
        cpu_frames = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
        cuda_frames = cpu_frames.to("cuda")
        video.write(cpu_frames)
        with pytest.raises(RuntimeError, match="same device"):
            video.write(cuda_frames)

    @pytest.mark.needs_cuda
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_samples_on_cuda_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        audio = enc.add_audio(sample_rate=44100, num_channels=1)
        enc.open(**open_kwargs)
        cuda_samples = torch.randn(1, 1000, device="cuda")
        with pytest.raises(RuntimeError, match="samples must be on CPU, got cuda"):
            audio.write(cuda_samples)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize(
        "device", ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda))
    )
    def test_write_frames_without_open_errors(self, tmp_path, method, device):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=30.0, device=device)
        frames = torch.randint(0, 256, (5, 3, 64, 64), dtype=torch.uint8, device=device)
        with pytest.raises(
            RuntimeError, match="Call open\\(\\) before addFrames\\(\\)"
        ):
            video.write(frames)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_samples_without_open_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        audio = enc.add_audio(sample_rate=44100, num_channels=1)
        samples = torch.randn(1, 1000)
        with pytest.raises(
            RuntimeError, match="Call open\\(\\) before addSamples\\(\\)"
        ):
            audio.write(samples)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_samples_mismatched_channels_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        audio = enc.add_audio(sample_rate=44100, num_channels=1)
        enc.open(**open_kwargs)
        samples = torch.randn(2, 1000)  # 2 channels but stream expects 1
        with pytest.raises(RuntimeError, match="Expected 1 channels, got 2"):
            audio.write(samples)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_open_without_stream_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        with pytest.raises(
            RuntimeError,
            match="Call addVideoStream\\(\\) or addAudioStream\\(\\) before open\\(\\)",
        ):
            enc.open(**open_kwargs)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_open_twice_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.add_video(height=64, width=64, frame_rate=30.0)
        enc.open(**open_kwargs)
        with pytest.raises(RuntimeError, match="open\\(\\) was already called"):
            enc.open(**open_kwargs)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_invalid_bit_rate_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.add_audio(sample_rate=44100, num_channels=2, bit_rate=-1)
        with pytest.raises(RuntimeError, match="bit_rate=-1 must be >= 0"):
            enc.open(**open_kwargs)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_and_encode_samples(self, tmp_path, method):
        source_audio = AudioDecoder(str(SINE_MONO_S32.path)).get_all_samples()
        samples = source_audio.data
        sample_rate = source_audio.sample_rate
        num_channels = samples.shape[0]

        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        audio = enc.add_audio(sample_rate=sample_rate, num_channels=num_channels)
        enc.open(**open_kwargs)
        chunk_lengths = [1, 50, 1000, 0, 25]
        offset = 0
        for length in chunk_lengths:
            audio.write(samples[:, offset : offset + length])
            offset += length
        audio.write(samples[:, offset:])
        enc.close()

        decoded = AudioDecoder(
            self._get_decoder_source(encoder_output)
        ).get_all_samples()
        assert decoded.data.shape[0] == num_channels
        assert decoded.sample_rate == sample_rate
        torch.testing.assert_close(decoded.data, samples, atol=1e-4, rtol=0)

    @pytest.mark.parametrize(
        "format",
        (
            "mp4",
            pytest.param(
                "mkv",
                marks=pytest.mark.skipif(
                    ffmpeg_major_version < 6,
                    reason="Default audio codec for MKV has low accuracy on older FFmpeg versions.",
                ),
            ),
            "mov",
        ),
    )
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_and_video_and_encode(self, tmp_path, format, method):
        source_video_decoder = VideoDecoder(str(NASA_VIDEO.path))
        source_frames = source_video_decoder.get_frames_in_range(
            start=0, stop=len(source_video_decoder)
        ).data

        source_audio = AudioDecoder(str(NASA_AUDIO_MP3_44100.path)).get_all_samples()
        source_samples = source_audio.data
        sample_rate = source_audio.sample_rate

        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=source_video_decoder.metadata.average_fps,
            pixel_format="yuv444p",
            crf=0,
        )
        audio = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=source_samples.shape[0],
        )
        enc.open(**open_kwargs)
        half_frames = source_frames.shape[0] // 2
        half_samples = source_samples.shape[1] // 2
        video.write(source_frames[:half_frames])
        audio.write(source_samples[:, :half_samples])
        video.write(source_frames[half_frames:])
        audio.write(source_samples[:, half_samples:])
        enc.close()

        source = self._get_decoder_source(encoder_output)

        decoded_video_decoder = VideoDecoder(source)
        decoded_frames = decoded_video_decoder.get_frames_in_range(
            start=0, stop=len(decoded_video_decoder)
        ).data
        assert_tensor_close_on_at_least(
            decoded_frames, source_frames, percentage=99, atol=2
        )

        audio_decoder = AudioDecoder(source)
        decoded_audio = audio_decoder.get_all_samples()
        assert decoded_audio.sample_rate == sample_rate
        assert decoded_audio.data.shape[0] == source_samples.shape[0]
        # Codecs for lossy audio formats (not WAV or FLAC) can add padding which causes
        # sample count to differ, so we only compare the smaller sample count.
        # TODO MultiStreamEncoder: The previous AudioEncoder didn't need
        # padding after introducing a FIFO. Investigate why this is needed.
        num_samples_to_compare = min(
            decoded_audio.data.shape[1], source_samples.shape[1]
        )
        assert_tensor_close_on_at_least(
            decoded_audio.data[:, :num_samples_to_compare],
            source_samples[:, :num_samples_to_compare],
            percentage=96 if format == "mkv" else 99,
            atol=0.1 if format == "mkv" else 0.01,
        )

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_multiple_video_streams_and_audio(self, tmp_path, method):
        source_frames_big = torch.randint(0, 256, (5, 3, 256, 256), dtype=torch.uint8)
        source_frames_small = torch.randint(0, 256, (8, 3, 128, 128), dtype=torch.uint8)

        source_audio = AudioDecoder(str(NASA_AUDIO_MP3_44100.path)).get_all_samples()
        source_samples_stereo = source_audio.data
        sample_rate = source_audio.sample_rate
        source_samples_mono = source_samples_stereo[:1]

        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video_big = enc.add_video(
            height=256, width=256, frame_rate=10.0, pixel_format="yuv444p", crf=0
        )
        video_small = enc.add_video(
            height=128, width=128, frame_rate=25.0, pixel_format="yuv444p", crf=0
        )
        audio_stereo = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=2,
        )
        audio_mono = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=1,
        )
        enc.open(**open_kwargs)
        video_big.write(source_frames_big[:3])
        video_small.write(source_frames_small[:4])
        audio_stereo.write(
            source_samples_stereo[:, : source_samples_stereo.shape[1] // 2]
        )
        audio_mono.write(source_samples_mono[:, : source_samples_mono.shape[1] // 2])
        video_big.write(source_frames_big[3:])
        video_small.write(source_frames_small[4:])
        audio_stereo.write(
            source_samples_stereo[:, source_samples_stereo.shape[1] // 2 :]
        )
        audio_mono.write(source_samples_mono[:, source_samples_mono.shape[1] // 2 :])
        enc.close()

        source = self._get_decoder_source(encoder_output)

        decoded_big = VideoDecoder(source, stream_index=0)
        assert len(decoded_big) == 5
        decoded_big_frames = decoded_big.get_frames_in_range(
            start=0, stop=len(decoded_big)
        ).data
        assert decoded_big_frames.shape == (5, 3, 256, 256)
        assert_tensor_close_on_at_least(
            decoded_big_frames, source_frames_big, percentage=99, atol=2
        )

        decoded_small = VideoDecoder(source, stream_index=1)
        assert len(decoded_small) == 8
        decoded_small_frames = decoded_small.get_frames_in_range(
            start=0, stop=len(decoded_small)
        ).data
        assert decoded_small_frames.shape == (8, 3, 128, 128)
        assert_tensor_close_on_at_least(
            decoded_small_frames, source_frames_small, percentage=99, atol=2
        )

        # stream_index is absolute: 0, 1 are video; 2, 3 are audio
        decoded_stereo = AudioDecoder(source, stream_index=2).get_all_samples()
        assert decoded_stereo.sample_rate == sample_rate
        assert decoded_stereo.data.shape[0] == 2
        num_samples_to_compare = min(
            decoded_stereo.data.shape[1], source_samples_stereo.shape[1]
        )
        assert_tensor_close_on_at_least(
            decoded_stereo.data[:, :num_samples_to_compare],
            source_samples_stereo[:, :num_samples_to_compare],
            percentage=98,
            atol=0.01,
        )

        decoded_mono = AudioDecoder(source, stream_index=3).get_all_samples()
        assert decoded_mono.sample_rate == sample_rate
        assert decoded_mono.data.shape[0] == 1
        num_samples_to_compare = min(
            decoded_mono.data.shape[1], source_samples_mono.shape[1]
        )
        assert_tensor_close_on_at_least(
            decoded_mono.data[:, :num_samples_to_compare],
            source_samples_mono[:, :num_samples_to_compare],
            percentage=98,
            atol=0.01,
        )

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_output_num_channels(self, tmp_path, method):
        sample_rate = 44_100
        source_stereo = torch.rand(2, 1_000)
        source_mono = torch.rand(1, 1_000)

        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mkv")
        # Stream 0: stereo input, mono output
        audio_stereo_to_mono = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=2,
            output_num_channels=1,
        )
        # Stream 1: mono input, stereo output
        audio_mono_to_stereo = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=1,
            output_num_channels=2,
        )
        # Stream 2: stereo input, no output_num_channels (should stay stereo)
        audio_passthrough = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=2,
        )
        enc.open(**open_kwargs)
        audio_stereo_to_mono.write(source_stereo)
        audio_mono_to_stereo.write(source_mono)
        audio_passthrough.write(source_stereo)
        enc.close()

        source = self._get_decoder_source(encoder_output)

        decoded_0 = AudioDecoder(source, stream_index=0).get_all_samples()
        assert decoded_0.data.shape[0] == 1

        decoded_1 = AudioDecoder(source, stream_index=1).get_all_samples()
        assert decoded_1.data.shape[0] == 2

        decoded_2 = AudioDecoder(source, stream_index=2).get_all_samples()
        assert decoded_2.data.shape[0] == 2

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_output_sample_rate(self, tmp_path, method):
        in_sample_rate = 44_100
        source_samples = torch.rand(1, 10_000)

        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mkv")
        # Stream 0: 44100 -> 48000
        audio_upsample = enc.add_audio(
            sample_rate=in_sample_rate,
            num_channels=1,
            output_sample_rate=48_000,
        )
        # Stream 1: 44100 -> 32000
        audio_downsample = enc.add_audio(
            sample_rate=in_sample_rate,
            num_channels=1,
            output_sample_rate=32_000,
        )
        # Stream 2: 44100 -> no conversion (stays 44100)
        audio_passthrough = enc.add_audio(
            sample_rate=in_sample_rate,
            num_channels=1,
        )
        enc.open(**open_kwargs)
        audio_upsample.write(source_samples)
        audio_downsample.write(source_samples)
        audio_passthrough.write(source_samples)
        enc.close()

        source = self._get_decoder_source(encoder_output)

        decoded_0 = AudioDecoder(source, stream_index=0).get_all_samples()
        assert decoded_0.sample_rate == 48_000

        decoded_1 = AudioDecoder(source, stream_index=1).get_all_samples()
        assert decoded_1.sample_rate == 32_000

        decoded_2 = AudioDecoder(source, stream_index=2).get_all_samples()
        assert decoded_2.sample_rate == 44_100

    @staticmethod
    def _get_video_metadata(file_path, fields):
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
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
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

    @needs_ffmpeg_cli
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
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
    def test_codec_parameter_utilized(self, tmp_path, method, format, codec_spec):
        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        video = enc.add_video(height=64, width=64, frame_rate=30.0, codec=codec_spec)
        enc.open(**open_kwargs)
        video.write(torch.zeros((10, 3, 64, 64), dtype=torch.uint8))
        enc.close()

        if method == "to_file_like":
            return
        actual = self._get_video_metadata(encoder_output, fields=["codec_name"])[
            "codec_name"
        ]
        assert actual == codec_spec

    @needs_ffmpeg_cli
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
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
    def test_codec_spec_vs_impl_equivalence(
        self, tmp_path, method, codec_spec, codec_impl
    ):
        frames = torch.randint(0, 256, (10, 3, 64, 64), dtype=torch.uint8)

        def encode_with_codec(codec, suffix):
            sub = tmp_path / suffix
            sub.mkdir(exist_ok=True)
            enc, encoder_output, open_kwargs = self._create_encoder(method, sub, "mp4")
            video = enc.add_video(height=64, width=64, frame_rate=30.0, codec=codec)
            enc.open(**open_kwargs)
            video.write(frames)
            enc.close()
            return encoder_output

        spec_output = encode_with_codec(codec_spec, "spec")
        impl_output = encode_with_codec(codec_impl, "impl")

        spec_decoded = (
            VideoDecoder(self._get_decoder_source(spec_output))
            .get_frames_in_range(start=0, stop=10)
            .data
        )
        impl_decoded = (
            VideoDecoder(self._get_decoder_source(impl_output))
            .get_frames_in_range(start=0, stop=10)
            .data
        )
        torch.testing.assert_close(spec_decoded, impl_decoded, rtol=0, atol=0)

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
        enc = StreamingEncoder()
        dest = str(tmp_path / "output.mp4")
        video = enc.add_video(
            height=64,
            width=64,
            frame_rate=30.0,
            extra_options={
                "profile": profile,
                "colorspace": colorspace,
                "color_range": color_range,
            },
        )
        enc.open(dest=dest)
        video.write(torch.zeros((5, 3, 64, 64), dtype=torch.uint8))
        enc.close()

        metadata = self._get_video_metadata(
            dest, fields=["profile", "color_space", "color_range"]
        )
        expected_profile = "constrained baseline" if profile == "baseline" else profile
        assert metadata["profile"].lower() == expected_profile
        assert metadata["color_space"] == colorspace
        assert metadata["color_range"] == color_range

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("crf", [23, 23.5, -0.9])
    def test_crf_valid_values(self, method, crf, tmp_path):
        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=30.0, crf=crf)
        enc.open(**open_kwargs)
        video.write(torch.zeros((5, 3, 64, 64), dtype=torch.uint8))
        enc.close()

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("bit_rate", [None, 0, 44100, 999999999])
    def test_audio_bit_rate_positive_values(self, method, bit_rate, tmp_path):
        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        audio = enc.add_audio(sample_rate=44100, num_channels=1, bit_rate=bit_rate)
        enc.open(**open_kwargs)
        audio.write(torch.randn(1, 1000))
        enc.close()

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("format", ["wav", "mp3", "flac"])
    def test_multiple_audio_formats(self, method, format, tmp_path):
        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        audio = enc.add_audio(sample_rate=44100, num_channels=1)
        enc.open(**open_kwargs)
        audio.write(torch.randn(1, 10_000))
        enc.close()

        decoded = AudioDecoder(
            self._get_decoder_source(encoder_output)
        ).get_all_samples()
        assert decoded.sample_rate == 44100
        assert decoded.data.shape[0] == 1

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_non_integer_frame_rate(self, method, tmp_path):
        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=29.97)
        enc.open(**open_kwargs)
        video.write(torch.zeros((10, 3, 64, 64), dtype=torch.uint8))
        enc.close()

        decoded_decoder = VideoDecoder(self._get_decoder_source(encoder_output))
        assert abs(decoded_decoder.metadata.average_fps - 29.97) < 0.01

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_preset_parameter(self, method, tmp_path):
        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=30.0, preset="ultrafast")
        enc.open(**open_kwargs)
        video.write(torch.zeros((5, 3, 64, 64), dtype=torch.uint8))
        enc.close()

        decoded = (
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=5)
            .data
        )
        assert decoded.shape == (5, 3, 64, 64)
