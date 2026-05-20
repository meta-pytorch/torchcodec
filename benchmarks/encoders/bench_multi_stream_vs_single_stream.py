"""
Basic benchmark aimed at validating that the new Encoder class isn't slower than
the existing VideoEncoder and AudioEncoder.
"""

import os
import tempfile
from time import perf_counter_ns

import torch

from torchcodec.encoders import AudioEncoder, Encoder, VideoEncoder


def bench(f, *args, num_exp=100, warmup=0, **kwargs):
    for _ in range(warmup):
        f(*args, **kwargs)

    times = []
    for _ in range(num_exp):
        start = perf_counter_ns()
        f(*args, **kwargs)
        end = perf_counter_ns()
        times.append(end - start)
    return torch.tensor(times).float()


def report_stats(times, unit="ms"):
    mul = {
        "ns": 1,
        "µs": 1e-3,
        "ms": 1e-6,
        "s": 1e-9,
    }[unit]
    times = times * mul
    std = times.std().item()
    med = times.median().item()
    print(f"{med = :.2f}{unit} +- {std:.2f}")
    return med


# --- Config ---
height, width = 256, 256
frame_rate = 30
num_channels = 2
sample_rate = 48000
durations_s = [3, 10]


def encode_video_with_video_encoder(frames, dest):
    encoder = VideoEncoder(frames, frame_rate=frame_rate)
    encoder.to_file(dest)


def encode_video_with_streaming_encoder(frames, dest):
    encoder = Encoder()
    video_stream = encoder.add_video(
        height=frames.shape[2],
        width=frames.shape[3],
        frame_rate=frame_rate,
        device=str(frames.device),
    )
    with encoder.open_file(dest):
        video_stream.add_frames(frames)


def encode_audio_with_audio_encoder(samples, dest):
    encoder = AudioEncoder(samples, sample_rate=sample_rate)
    encoder.to_file(dest)


def encode_audio_with_streaming_encoder(samples, dest):
    encoder = Encoder()
    audio_stream = encoder.add_audio(sample_rate=sample_rate, num_channels=num_channels)
    with encoder.open_file(dest):
        audio_stream.add_samples(samples)


if __name__ == "__main__":
    tmpdir = tempfile.mkdtemp()

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for duration_s in durations_s:
        print(f"\n{'=' * 50}")
        print(f"Duration: {duration_s}s")
        print(f"{'=' * 50}")

        num_frames = frame_rate * duration_s
        video_frames = torch.randint(
            0, 256, (num_frames, 3, height, width), dtype=torch.uint8
        )

        num_samples = sample_rate * duration_s
        audio_samples = torch.randn(num_channels, num_samples)

        # --- Video benchmarks ---
        for device in devices:
            frames = video_frames.to(device)
            dest = os.path.join(tmpdir, "video.mp4")

            print(f"\n--- Video encoding ({device}) ---")

            print("VideoEncoder:    ", end="")
            times = bench(
                encode_video_with_video_encoder, frames, dest, num_exp=10, warmup=2
            )
            report_stats(times)

            print("StreamingEncoder:", end="")
            times = bench(
                encode_video_with_streaming_encoder, frames, dest, num_exp=10, warmup=2
            )
            report_stats(times)

        # --- Audio benchmarks ---
        print("\n--- Audio encoding (cpu) ---")
        dest = os.path.join(tmpdir, "audio.wav")

        print("AudioEncoder:    ", end="")
        times = bench(
            encode_audio_with_audio_encoder, audio_samples, dest, num_exp=10, warmup=2
        )
        report_stats(times)

        print("StreamingEncoder:", end="")
        times = bench(
            encode_audio_with_streaming_encoder,
            audio_samples,
            dest,
            num_exp=10,
            warmup=2,
        )
        report_stats(times)
