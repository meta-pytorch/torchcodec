import io
import os
import subprocess
from time import perf_counter_ns

import librosa
import torch

import torchcodec
from torchcodec.decoders import AudioDecoder, WavDecoder


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


AUDIO_DIR = "/tmp/librosa_bench_files"

# For .wav we decode with torchcodec's WavDecoder, for .mp3 with AudioDecoder.
FORMATS = {
    "wav": ("pcm_s16le", "wav"),
    "mp3": ("libmp3lame", "mp3"),
}

DURATIONS = {
    "10s": 10,
    "30s": 30,
    "1min": 60,
}

# The ffmpeg-generated sine inputs are at 44100 Hz, so 16kHz is a downsample
# and 48kHz is an upsample.
RESAMPLE_RATES = {
    "downsample 16kHz": 16000,
    "upsample 48kHz": 48000,
}


def generate_files():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    for fmt_name, (codec, ext) in FORMATS.items():
        for dur_name, dur_seconds in DURATIONS.items():
            path = os.path.join(AUDIO_DIR, f"{fmt_name}_{dur_name}.{ext}")
            if os.path.exists(path):
                continue
            print(f"Generating {path} ...")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"sine=frequency=440:duration={dur_seconds}",
                    "-c:a",
                    codec,
                    path,
                ],
                check=True,
                capture_output=True,
            )


def decode_wav_torchcodec(raw_bytes):
    return WavDecoder(raw_bytes).get_all_samples().data


def decode_audio_torchcodec(raw_bytes):
    return AudioDecoder(raw_bytes).get_all_samples().data


def decode_librosa(raw_bytes, sample_rate=None):
    # sample_rate=None -> decode at the file's native sample rate (no resampling).
    # When resampling, we explicitly use soxr (librosa's default res_type since 0.10).
    data, sr = librosa.load(
        io.BytesIO(raw_bytes), sr=sample_rate, mono=False, res_type="soxr_hq"
    )
    return data


def decode_resample_torchcodec(raw_bytes, sample_rate):
    # WavDecoder does not support resampling, so AudioDecoder is used for both
    # wav and mp3 in the resampling benchmark.
    return AudioDecoder(raw_bytes, sample_rate=sample_rate).get_all_samples().data


def main():
    print(f"torchcodec: {torchcodec.__version__}")
    print(f"librosa:    {librosa.__version__}")
    generate_files()

    results = []

    for fmt_name, (_, ext) in FORMATS.items():
        for dur_name in DURATIONS:
            path = os.path.join(AUDIO_DIR, f"{fmt_name}_{dur_name}.{ext}")
            with open(path, "rb") as f:
                raw_bytes = f.read()
            num_exp = 30 if dur_name == "1min" else 100

            tc_decode = (
                decode_wav_torchcodec
                if fmt_name == "wav"
                else decode_audio_torchcodec
            )
            tc_name = "WavDecoder" if fmt_name == "wav" else "AudioDecoder"

            print(f"\n=== {fmt_name} / {dur_name} ({path}) ===")

            print(f"  torchcodec ({tc_name}): ", end="")
            tc_times = bench(tc_decode, raw_bytes, num_exp=num_exp, warmup=2)
            tc_med = report_stats(tc_times)

            print("  librosa (soxr): ", end="")
            lr_times = bench(decode_librosa, raw_bytes, num_exp=num_exp, warmup=2)
            lr_med = report_stats(lr_times)

            results.append((fmt_name, dur_name, tc_name, tc_med, lr_med))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'format':<8} {'duration':<10} {'decoder':<14} "
        f"{'torchcodec (ms)':>16} {'librosa (ms)':>14} {'librosa/tc':>12}"
    )
    print("-" * 80)
    for fmt_name, dur_name, tc_name, tc_med, lr_med in results:
        ratio = lr_med / tc_med if tc_med > 0 else float("inf")
        print(
            f"{fmt_name:<8} {dur_name:<10} {tc_name:<14} "
            f"{tc_med:>16.2f} {lr_med:>14.2f} {ratio:>11.2f}x"
        )


def bench_resampling():
    print("\n" + "=" * 80)
    print("RESAMPLING (torchcodec AudioDecoder vs librosa soxr_hq)")
    print("=" * 80)

    results = []
    for fmt_name, (_, ext) in FORMATS.items():
        for dur_name in DURATIONS:
            path = os.path.join(AUDIO_DIR, f"{fmt_name}_{dur_name}.{ext}")
            with open(path, "rb") as f:
                raw_bytes = f.read()
            num_exp = 30 if dur_name == "1min" else 100

            for rate_name, target_sr in RESAMPLE_RATES.items():
                print(f"\n=== {fmt_name} / {dur_name} / {rate_name} ({target_sr}) ===")

                print("  torchcodec (AudioDecoder): ", end="")
                tc_times = bench(
                    decode_resample_torchcodec,
                    raw_bytes,
                    target_sr,
                    num_exp=num_exp,
                    warmup=2,
                )
                tc_med = report_stats(tc_times)

                print("  librosa (soxr_hq): ", end="")
                lr_times = bench(
                    decode_librosa,
                    raw_bytes,
                    target_sr,
                    num_exp=num_exp,
                    warmup=2,
                )
                lr_med = report_stats(lr_times)

                results.append((fmt_name, dur_name, rate_name, tc_med, lr_med))

    print("\n" + "=" * 80)
    print("RESAMPLING SUMMARY")
    print("=" * 80)
    print(
        f"{'format':<8} {'duration':<10} {'resample':<18} "
        f"{'torchcodec (ms)':>16} {'librosa (ms)':>14} {'librosa/tc':>12}"
    )
    print("-" * 80)
    for fmt_name, dur_name, rate_name, tc_med, lr_med in results:
        ratio = lr_med / tc_med if tc_med > 0 else float("inf")
        print(
            f"{fmt_name:<8} {dur_name:<10} {rate_name:<18} "
            f"{tc_med:>16.2f} {lr_med:>14.2f} {ratio:>11.2f}x"
        )


if __name__ == "__main__":
    main()
    bench_resampling()
