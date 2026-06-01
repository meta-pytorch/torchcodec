import io
import os
import subprocess
from time import perf_counter_ns

import soundfile as sf
import torch

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


WAV_DIR = "/tmp/wav_files"

FORMATS = {
    "u8": ("pcm_u8", "int16"),
    "s16": ("pcm_s16le", "int16"),
    "s24": ("pcm_s24le", "int32"),
    "s32": ("pcm_s32le", "int32"),
    "float32": ("pcm_f32le", "float32"),
    "float64": ("pcm_f64le", "float64"),
}

DURATIONS = {
    "10s": 10,
    "5min": 300,
}


def generate_wav_files():
    os.makedirs(WAV_DIR, exist_ok=True)
    for fmt_name, (codec, _) in FORMATS.items():
        for dur_name, dur_seconds in DURATIONS.items():
            path = os.path.join(WAV_DIR, f"{fmt_name}_{dur_name}.wav")
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


def decode_torchcodec(raw_bytes):
    decoder = WavDecoder(raw_bytes)
    return decoder.get_all_samples()


def decode_audio_decoder(raw_bytes):
    decoder = AudioDecoder(raw_bytes)
    return decoder.get_all_samples().data


def decode_soundfile(raw_bytes):
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    return data


def decode_soundfile_native(raw_bytes, native_dtype):
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype=native_dtype)
    return data


def validate_results():
    print("Validating WavDecoder vs AudioDecoder outputs...")
    for fmt_name in FORMATS:
        for dur_name in DURATIONS:
            path = os.path.join(WAV_DIR, f"{fmt_name}_{dur_name}.wav")
            with open(path, "rb") as f:
                raw_bytes = f.read()
            wav_out = decode_torchcodec(raw_bytes)
            audio_out = decode_audio_decoder(raw_bytes)
            torch.testing.assert_close(wav_out.data, audio_out.data, atol=0, rtol=0)
            print(f"  {fmt_name}/{dur_name}: OK")
    print("All validations passed!\n")


def main():
    generate_wav_files()
    validate_results()

    results = []

    for fmt_name, (_, native_dtype) in FORMATS.items():
        for dur_name in DURATIONS:
            path = os.path.join(WAV_DIR, f"{fmt_name}_{dur_name}.wav")
            with open(path, "rb") as f:
                raw_bytes = f.read()
            num_exp = 10 if dur_name == "5min" else 100

            print(f"\n=== {fmt_name} / {dur_name} ({path}) ===")

            print("  WavDecoder: ", end="")
            tc_times = bench(decode_torchcodec, raw_bytes, num_exp=num_exp, warmup=2)
            tc_med = report_stats(tc_times)

            print("  AudioDecoder: ", end="")
            ad_times = bench(decode_audio_decoder, raw_bytes, num_exp=num_exp, warmup=2)
            ad_med = report_stats(ad_times)

            print("  soundfile (dtype=float32): ", end="")
            sf_times = bench(decode_soundfile, raw_bytes, num_exp=num_exp, warmup=2)
            sf_med = report_stats(sf_times)

            print(f"  soundfile (dtype={native_dtype}): ", end="")
            sfn_times = bench(
                decode_soundfile_native,
                raw_bytes,
                native_dtype,
                num_exp=num_exp,
                warmup=2,
            )
            sfn_med = report_stats(sfn_times)

            results.append((fmt_name, dur_name, tc_med, ad_med, sf_med, sfn_med))

    print("\n" + "=" * 155)
    print("SUMMARY")
    print("=" * 155)
    print(
        f"{'format':<10} {'duration':<10} "
        f"{'WavDec (ms)':>12} {'AudioDec (ms)':>14} {'sndfile f32 (ms)':>17} {'sndfile native (ms)':>20} "
        f"{'AudioDec/WavDec':>16} {'sndfile f32/WavDec':>19} {'sndfile nat/WavDec':>19}"
    )
    print("-" * 155)
    for fmt_name, dur_name, tc_med, ad_med, sf_med, sfn_med in results:
        audio_over_wav = ad_med / tc_med if tc_med > 0 else float("inf")
        sf_over_wav = sf_med / tc_med if tc_med > 0 else float("inf")
        sfn_over_wav = sfn_med / tc_med if tc_med > 0 else float("inf")
        print(
            f"{fmt_name:<10} {dur_name:<10} "
            f"{tc_med:>12.2f} {ad_med:>14.2f} {sf_med:>17.2f} {sfn_med:>20.2f} "
            f"{audio_over_wav:>15.2f}x {sf_over_wav:>18.2f}x {sfn_over_wav:>18.2f}x"
        )


def bench_input_types():
    print("\n" + "=" * 100)
    print("FILE vs FILE-LIKE vs BYTES")
    print("=" * 100)
    print(
        f"{'format':<10} {'duration':<10} "
        f"{'file (ms)':>12} {'file-like (ms)':>16} {'bytes (ms)':>12} "
        f"{'flike/file':>12} {'bytes/file':>12}"
    )
    print("-" * 100)

    for fmt_name in FORMATS:
        for dur_name in DURATIONS:
            path = os.path.join(WAV_DIR, f"{fmt_name}_{dur_name}.wav")
            num_exp = 10 if dur_name == "5min" else 100

            with open(path, "rb") as f:
                raw_bytes = f.read()

            def decode_file():
                return WavDecoder(path).get_all_samples()

            def decode_filelike():
                with open(path, "rb") as f:
                    return WavDecoder(f).get_all_samples()

            def decode_bytes():
                return WavDecoder(raw_bytes).get_all_samples()

            file_med = (
                bench(decode_file, num_exp=num_exp, warmup=2).median().item() * 1e-6
            )
            flike_med = (
                bench(decode_filelike, num_exp=num_exp, warmup=2).median().item() * 1e-6
            )
            bytes_med = (
                bench(decode_bytes, num_exp=num_exp, warmup=2).median().item() * 1e-6
            )

            flike_ratio = flike_med / file_med if file_med > 0 else float("inf")
            bytes_ratio = bytes_med / file_med if file_med > 0 else float("inf")

            print(
                f"{fmt_name:<10} {dur_name:<10} "
                f"{file_med:>12.2f} {flike_med:>16.2f} {bytes_med:>12.2f} "
                f"{flike_ratio:>11.2f}x {bytes_ratio:>11.2f}x"
            )


if __name__ == "__main__":
    main()
    bench_input_types()
