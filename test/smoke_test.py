from pathlib import Path

import pytest
import torch

from test.utils import assert_tensor_close_on_at_least

from torchcodec import ffmpeg_major_version
from torchcodec._frame import AudioSamples, Frame, FrameBatch
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torchcodec.encoders import AudioEncoder, VideoEncoder
from torchcodec.encoders._multi_stream_encoder import StreamingEncoder


@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(0)


NUM_FRAMES = 10
HEIGHT = 256
WIDTH = 256
FRAME_RATE = 30
NUM_AUDIO_CHANNELS = 2
SAMPLE_RATE = 16_000
NUM_SAMPLES = 10_000


def _make_video_file(tmp_path, pixel_format="yuv444p", **encoder_kwargs):
    frames = torch.randint(0, 256, (NUM_FRAMES, 3, HEIGHT, WIDTH), dtype=torch.uint8)
    path = tmp_path / "test.mp4"
    VideoEncoder(frames, frame_rate=FRAME_RATE).to_file(
        path, pixel_format=pixel_format, crf=0, **encoder_kwargs
    )
    return path, frames


def _make_audio_file(tmp_path, *, format="wav"):
    samples = torch.rand(NUM_AUDIO_CHANNELS, NUM_SAMPLES) * 2 - 1
    path = tmp_path / f"test.{format}"
    AudioEncoder(samples, sample_rate=SAMPLE_RATE).to_file(path)
    return path, samples


def _get_devices():
    return (
        "cpu",
        pytest.param("cuda", marks=pytest.mark.needs_cuda),
    )


# On CPU we encode with yuv444p (near-lossless with crf=0) so we can compare
# decoded frames directly against the source.
# On CUDA we encode with yuv420p instead, because yuv444p would trigger a CPU
# fallback. Since yuv420p is lossy on random data we can't compare against
# source, so we compare CUDA-decoded output against CPU-decoded output instead.
def _make_decoder_and_ref(tmp_path, device):
    """Returns (decoder, ref_decoder_or_none, source_frames_or_none).

    On CPU: returns (cpu_decoder, None, source_frames)
    On CUDA: returns (cuda_decoder, cpu_decoder, None)
    """
    if device == "cpu":
        path, source_frames = _make_video_file(tmp_path, pixel_format="yuv444p")
        return VideoDecoder(path, device="cpu"), None, source_frames
    else:
        path, _ = _make_video_file(tmp_path, pixel_format="yuv420p")
        return (
            VideoDecoder(path, device="cuda"),
            VideoDecoder(path, device="cpu"),
            None,
        )


def _assert_frames_close(decoded, *, ref_decoded=None, source=None, device):
    """Assert decoded frames are close to reference.

    On CPU, compares against source frames (near-lossless yuv444p roundtrip).
    On CUDA, compares against CPU-decoded frames (both from same yuv420p file).
    """
    actual = decoded.cpu() if device != "cpu" else decoded
    if device == "cpu":
        assert source is not None
        torch.testing.assert_close(actual, source, atol=2, rtol=0)
    else:
        assert ref_decoded is not None
        assert_tensor_close_on_at_least(
            actual, ref_decoded.cpu(), percentage=95, atol=3
        )


class TestVideoDecoder:
    @pytest.mark.parametrize("device", _get_devices())
    def test_basics(self, tmp_path, device):
        path, source_frames = _make_video_file(tmp_path)
        decoder = VideoDecoder(path, device=device)

        assert len(decoder) == NUM_FRAMES
        assert decoder.metadata.height == HEIGHT
        assert decoder.metadata.width == WIDTH

    @pytest.mark.parametrize("device", _get_devices())
    def test_get_frame_at(self, tmp_path, device):
        decoder, ref_decoder, source_frames = _make_decoder_and_ref(tmp_path, device)
        frame = decoder.get_frame_at(0)
        assert isinstance(frame, Frame)
        assert frame.data.shape == (3, HEIGHT, WIDTH)
        assert frame.data.dtype == torch.uint8
        _assert_frames_close(
            frame.data,
            ref_decoded=ref_decoder.get_frame_at(0).data if ref_decoder else None,
            source=source_frames[0] if source_frames is not None else None,
            device=device,
        )

    @pytest.mark.parametrize("device", _get_devices())
    def test_get_frames_in_range(self, tmp_path, device):
        decoder, ref_decoder, source_frames = _make_decoder_and_ref(tmp_path, device)
        batch = decoder.get_frames_in_range(start=0, stop=5)
        assert isinstance(batch, FrameBatch)
        assert batch.data.shape == (5, 3, HEIGHT, WIDTH)
        _assert_frames_close(
            batch.data,
            ref_decoded=(
                ref_decoder.get_frames_in_range(start=0, stop=5).data
                if ref_decoder
                else None
            ),
            source=source_frames[:5] if source_frames is not None else None,
            device=device,
        )

    @pytest.mark.parametrize("device", _get_devices())
    def test_get_frame_played_at(self, tmp_path, device):
        path, _ = _make_video_file(tmp_path)
        decoder = VideoDecoder(path, device=device)

        frame = decoder.get_frame_played_at(0.0)
        assert isinstance(frame, Frame)
        assert frame.data.shape == (3, HEIGHT, WIDTH)

    @pytest.mark.parametrize("device", _get_devices())
    def test_getitem(self, tmp_path, device):
        decoder, ref_decoder, source_frames = _make_decoder_and_ref(tmp_path, device)

        tensor = decoder[0]
        assert tensor.shape == (3, HEIGHT, WIDTH)
        _assert_frames_close(
            tensor,
            ref_decoded=ref_decoder[0] if ref_decoder else None,
            source=source_frames[0] if source_frames is not None else None,
            device=device,
        )

        tensors = decoder[2:5]
        assert tensors.shape == (3, 3, HEIGHT, WIDTH)
        _assert_frames_close(
            tensors,
            ref_decoded=ref_decoder[2:5] if ref_decoder else None,
            source=source_frames[2:5] if source_frames is not None else None,
            device=device,
        )

    @pytest.mark.parametrize("device", _get_devices())
    def test_get_all_frames(self, tmp_path, device):
        decoder, ref_decoder, source_frames = _make_decoder_and_ref(tmp_path, device)
        all_frames = decoder.get_all_frames()
        assert all_frames.data.shape == (NUM_FRAMES, 3, HEIGHT, WIDTH)
        _assert_frames_close(
            all_frames.data,
            ref_decoded=(ref_decoder.get_all_frames().data if ref_decoder else None),
            source=source_frames,
            device=device,
        )

    @pytest.mark.parametrize("device", _get_devices())
    def test_iteration(self, tmp_path, device):
        path, _ = _make_video_file(tmp_path)
        decoder = VideoDecoder(path, device=device)

        count = 0
        for frame in decoder:
            assert frame.shape == (3, HEIGHT, WIDTH)
            count += 1
        assert count == NUM_FRAMES


class TestAudioDecoder:
    def test_basics(self, tmp_path):
        path, source_samples = _make_audio_file(tmp_path)
        decoder = AudioDecoder(path)

        assert decoder.metadata.sample_rate == SAMPLE_RATE
        assert decoder.metadata.num_channels == NUM_AUDIO_CHANNELS

    def test_get_all_samples(self, tmp_path):
        path, source_samples = _make_audio_file(tmp_path)
        decoder = AudioDecoder(path)

        samples = decoder.get_all_samples()
        assert isinstance(samples, AudioSamples)
        assert samples.data.shape == (NUM_AUDIO_CHANNELS, NUM_SAMPLES)
        assert samples.sample_rate == SAMPLE_RATE
        assert samples.pts_seconds == 0.0
        assert samples.duration_seconds > 0
        torch.testing.assert_close(samples.data, source_samples, atol=1e-4, rtol=1e-3)

    def test_get_samples_played_in_range(self, tmp_path):
        path, source_samples = _make_audio_file(tmp_path)
        decoder = AudioDecoder(path)

        samples = decoder.get_samples_played_in_range(
            start_seconds=0.0, stop_seconds=0.1
        )
        assert isinstance(samples, AudioSamples)
        assert samples.data.shape[0] == NUM_AUDIO_CHANNELS
        expected_num_samples = int(0.1 * SAMPLE_RATE)
        assert abs(samples.data.shape[1] - expected_num_samples) <= 1

    def test_resample_on_decode(self, tmp_path):
        path, source_samples = _make_audio_file(tmp_path)

        target_sr = 8000
        decoder = AudioDecoder(path, sample_rate=target_sr, num_channels=1)

        samples = decoder.get_all_samples()
        assert samples.sample_rate == target_sr
        assert samples.data.shape[0] == 1


class TestVideoEncoder:
    def test_to_file(self, tmp_path):
        frames = torch.randint(0, 256, (5, 3, HEIGHT, WIDTH), dtype=torch.uint8)
        path = str(tmp_path / "out.mp4")
        VideoEncoder(frames, frame_rate=FRAME_RATE).to_file(path)
        assert Path(path).stat().st_size > 0

        decoder = VideoDecoder(path)
        assert len(decoder) == 5

    def test_to_tensor(self):
        frames = torch.randint(0, 256, (5, 3, HEIGHT, WIDTH), dtype=torch.uint8)
        encoded = VideoEncoder(frames, frame_rate=FRAME_RATE).to_tensor(format="mp4")
        assert encoded.dtype == torch.uint8
        assert encoded.ndim == 1
        assert len(encoded) > 0

    def test_roundtrip_lossless(self, tmp_path):
        frames = torch.randint(0, 256, (5, 3, HEIGHT, WIDTH), dtype=torch.uint8)
        path = str(tmp_path / "lossless.mp4")
        VideoEncoder(frames, frame_rate=FRAME_RATE).to_file(
            path, pixel_format="yuv444p", crf=0
        )
        decoder = VideoDecoder(path)
        decoded = decoder.get_all_frames()
        torch.testing.assert_close(decoded.data, frames, atol=2, rtol=0)


class TestAudioEncoder:
    def test_to_file_wav(self, tmp_path):
        samples = torch.rand(2, NUM_SAMPLES) * 2 - 1
        path = str(tmp_path / "out.wav")
        AudioEncoder(samples, sample_rate=SAMPLE_RATE).to_file(path)
        assert Path(path).stat().st_size > 0

        decoder = AudioDecoder(path)
        assert decoder.metadata.sample_rate == SAMPLE_RATE
        assert decoder.metadata.num_channels == 2
        decoded = decoder.get_all_samples()
        assert decoded.data.shape == (2, NUM_SAMPLES)
        torch.testing.assert_close(decoded.data, samples, atol=1e-4, rtol=1e-3)

    def test_to_tensor(self):
        samples = torch.rand(1, NUM_SAMPLES) * 2 - 1
        encoded = AudioEncoder(samples, sample_rate=SAMPLE_RATE).to_tensor(format="wav")
        assert encoded.dtype == torch.uint8
        assert encoded.ndim == 1
        assert len(encoded) > 0

        decoder = AudioDecoder(encoded)
        decoded = decoder.get_all_samples()
        assert decoded.data.shape == (1, NUM_SAMPLES)
        torch.testing.assert_close(decoded.data, samples, atol=1e-4, rtol=1e-3)

    def test_mono_1d_input(self, tmp_path):
        samples = torch.rand(NUM_SAMPLES) * 2 - 1
        path = str(tmp_path / "mono.wav")
        AudioEncoder(samples, sample_rate=SAMPLE_RATE).to_file(path)

        decoder = AudioDecoder(path)
        assert decoder.metadata.num_channels == 1
        decoded = decoder.get_all_samples()
        assert decoded.data.shape == (1, NUM_SAMPLES)
        torch.testing.assert_close(decoded.data[0], samples, atol=1e-4, rtol=1e-3)

    def test_resample_on_encode(self, tmp_path):
        samples = torch.rand(1, NUM_SAMPLES) * 2 - 1
        path = str(tmp_path / "resampled.wav")
        AudioEncoder(samples, sample_rate=SAMPLE_RATE).to_file(path, sample_rate=8000)
        decoder = AudioDecoder(path)
        assert decoder.metadata.sample_rate == 8000
        decoded = decoder.get_all_samples()
        assert decoded.data.shape[0] == 1
        expected_num_samples = int(NUM_SAMPLES * 8000 / SAMPLE_RATE)
        assert abs(decoded.data.shape[1] - expected_num_samples) <= 1


class TestStreamingEncoder:
    def test_video_and_audio_chunked(self, tmp_path):
        frames = torch.randint(
            0, 256, (NUM_FRAMES, 3, HEIGHT, WIDTH), dtype=torch.uint8
        )
        sr = 44100
        samples = torch.rand(NUM_AUDIO_CHANNELS, NUM_SAMPLES) * 2 - 1
        path = tmp_path / "av.mkv"

        enc = StreamingEncoder()
        video = enc.add_video(
            height=HEIGHT,
            width=WIDTH,
            frame_rate=FRAME_RATE,
            pixel_format="yuv444p",
            crf=0,
        )
        audio = enc.add_audio(sample_rate=sr, num_channels=NUM_AUDIO_CHANNELS)
        enc.open(dest=path)
        with enc:
            video.write(frames[:5])
            audio.write(samples[:, : NUM_SAMPLES // 2])
            video.write(frames[5:])
            audio.write(samples[:, NUM_SAMPLES // 2 :])

        video_dec = VideoDecoder(path)
        assert len(video_dec) == NUM_FRAMES
        decoded_frames = video_dec.get_all_frames()
        torch.testing.assert_close(decoded_frames.data, frames, atol=2, rtol=0)

        audio_dec = AudioDecoder(path)
        assert audio_dec.metadata.num_channels == NUM_AUDIO_CHANNELS
        assert audio_dec.metadata.sample_rate == sr
        decoded_samples = audio_dec.get_all_samples()
        assert decoded_samples.data.shape[0] == NUM_AUDIO_CHANNELS
        # TODO: validate audio on a mostly lossless codec?
        assert decoded_samples.sample_rate == sr

    @pytest.mark.needs_cuda
    def test_cuda_encoding(self, tmp_path):
        if ffmpeg_major_version == 4:
            pytest.skip("CUDA encoding not supported with FFmpeg 4")

        # Use smooth gradient data instead of random noise. Random data is
        # incompressible, causing very different artifacts between nvenc and
        # libx264.
        frames = torch.zeros(NUM_FRAMES, 3, HEIGHT, WIDTH, dtype=torch.uint8)
        for i in range(NUM_FRAMES):
            for c in range(3):
                vals = (
                    torch.linspace(0, 255, WIDTH).unsqueeze(0).expand(HEIGHT, -1)
                    + i * 20
                    + c * 80
                ) % 256
                frames[i, c] = vals.to(torch.uint8)
        frames = frames.cuda()

        cuda_path = tmp_path / "cuda.mp4"
        enc = StreamingEncoder()
        video = enc.add_video(
            height=HEIGHT,
            width=WIDTH,
            frame_rate=FRAME_RATE,
            device="cuda",
        )
        with enc.open(dest=cuda_path):
            video.write(frames)

        cpu_path = tmp_path / "cpu.mp4"
        cpu_enc = StreamingEncoder()
        cpu_video = cpu_enc.add_video(
            height=HEIGHT,
            width=WIDTH,
            frame_rate=FRAME_RATE,
        )
        with cpu_enc.open(dest=cpu_path):
            cpu_video.write(frames.cpu())

        cuda_decoded = VideoDecoder(cuda_path).get_all_frames()
        cpu_decoded = VideoDecoder(cpu_path).get_all_frames()
        assert cuda_decoded.data.shape == (NUM_FRAMES, 3, HEIGHT, WIDTH)
        if ffmpeg_major_version != 4:
            assert_tensor_close_on_at_least(
                cuda_decoded.data, cpu_decoded.data, percentage=95, atol=3
            )
