#!/usr/bin/env python3
import shutil
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter_ns

import pynvml
import torch
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

FRAME_RATE = 30
DEFAULT_VIDEO_PATH = "test/resources/nasa_13013.mp4"
# Alternatively, run this command to generate a longer test video:
#   ffmpeg -f lavfi -i testsrc2=duration=600:size=1280x720:rate=30 -c:v libx264 -pix_fmt yuv420p test/resources/testsrc2_10min.mp4


def bench(f, average_over=50, warmup=2, gpu_monitoring=False, **f_kwargs):
    for _ in range(warmup):
        f(**f_kwargs)

    times = []
    utilizations = []
    memory_usage = []

    for _ in range(average_over):
        start = perf_counter_ns()
        f(**f_kwargs)
        end = perf_counter_ns()
        times.append(end - start)

        if gpu_monitoring:
            util = pynvml.nvmlDeviceGetEncoderUtilization(handle)[0]
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = mem_info.used / (1_000_000)  # Convert bytes to MB
            utilizations.append(util)
            memory_usage.append(mem_used)

    times_tensor = torch.tensor(times).float()
    return times_tensor, {
        "utilization": torch.tensor(utilizations).float() if gpu_monitoring else None,
        "memory_used": torch.tensor(memory_usage).float() if gpu_monitoring else None,
    }


def report_stats(times, num_frames, nvenc_metrics=None, prefix="", unit="ms"):
    fps = num_frames * 1e9 / times.median()

    mul = {
        "ns": 1,
        "µs": 1e-3,
        "ms": 1e-6,
        "s": 1e-9,
    }[unit]
    unit_times = times * mul
    med = unit_times.median().item()
    max = unit_times.max().item()
    print(f"\n{prefix}   {med = :.2f} {unit}, {max = :.2f} {unit}, fps = {fps:.1f}")

    if nvenc_metrics is not None:
        mem_used_max = nvenc_metrics["memory_used"].max().item()
        mem_used_median = nvenc_metrics["memory_used"].median().item()
        util_max = nvenc_metrics["utilization"].max().item()

        print(
            f"GPU memory used:      med = {mem_used_median:.1f} MB, max = {mem_used_max:.1f} MB"
        )
        print(
            f"NVENC utilization:    med = {nvenc_metrics["utilization"].median():.1f}%,     max = {util_max:.1f}%"
        )


def encode_torchcodec(frames, output_path, device="cpu"):
    encoder = VideoEncoder(frames=frames, frame_rate=FRAME_RATE)
    if device == "cuda":
        encoder.to_file(dest=output_path, codec="h264_nvenc", extra_options={"qp": 0})
    else:
        encoder.to_file(dest=output_path, codec="libx264", crf=0)


def write_raw_frames(frames, raw_path):
    # Convert NCHW to NHWC for raw video format
    raw_frames = frames.permute(0, 2, 3, 1)
    with open(raw_path, "wb") as f:
        f.write(raw_frames.cpu().numpy().tobytes())


def encode_ffmpeg_cli(
    frames, raw_path, output_path, device="cpu", skip_write_frames=False
):
    # Write frames during benchmarking function by default unless skip_write_frames flag used
    if not skip_write_frames:
        write_raw_frames(frames, raw_path)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{frames.shape[3]}x{frames.shape[2]}",
        "-r",
        str(FRAME_RATE),
        "-i",
        raw_path,
        "-c:v",
        "h264_nvenc" if device == "cuda" else "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    ffmpeg_cmd.extend(["-qp", "0"] if device == "cuda" else ["-crf", "0"])
    ffmpeg_cmd.extend([str(output_path)])
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--path", type=str, help="Path to input video file", default=DEFAULT_VIDEO_PATH
    )
    parser.add_argument(
        "--average-over",
        type=int,
        default=30,
        help="Number of runs to average over",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to decode for benchmarking. By default, all frames will be decoded.",
    )
    parser.add_argument(
        "--skip-write-frames",
        action="store_true",
        help="Do not write raw frames in FFmpeg CLI benchmarks",
    )
    args = parser.parse_args()
    decoder = VideoDecoder(str(args.path))
    frames = decoder.get_frames_in_range(start=0, stop=args.max_frames).data

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("CUDA not available. GPU benchmarks will be skipped.")

    print(
        f"Benchmarking {len(frames)} frames from {Path(args.path).name} over {args.average_over} runs:"
    )
    gpu_frames = frames.cuda() if cuda_available else None
    print(
        f"Decoded {frames.shape[0]} frames of size {frames.shape[2]}x{frames.shape[3]}"
    )

    temp_dir = Path(tempfile.mkdtemp())
    raw_frames_path = temp_dir / "input_frames.raw"

    # If skip_write_frames is True, we will not benchmark the time it takes to write the frames.
    # Here, we still write the frames for FFmpeg to use!
    if args.skip_write_frames:
        write_raw_frames(frames, str(raw_frames_path))

    if cuda_available:
        # Benchmark torchcodec on GPU
        gpu_output = temp_dir / "torchcodec_gpu.mp4"
        times, nvenc_metrics = bench(
            encode_torchcodec,
            frames=gpu_frames,
            output_path=str(gpu_output),
            device="cuda",
            gpu_monitoring=True,
            average_over=args.average_over,
        )
        report_stats(
            times, frames.shape[0], nvenc_metrics, prefix="VideoEncoder on GPU"
        )
        # Benchmark FFmpeg CLI on GPU
        ffmpeg_gpu_output = temp_dir / "ffmpeg_gpu.mp4"
        times, nvenc_metrics = bench(
            encode_ffmpeg_cli,
            frames=gpu_frames,
            raw_path=str(raw_frames_path),
            output_path=str(ffmpeg_gpu_output),
            device="cuda",
            gpu_monitoring=True,
            skip_write_frames=args.skip_write_frames,
            average_over=args.average_over,
        )
        prefix = "FFmpeg CLI on GPU  "
        report_stats(times, frames.shape[0], nvenc_metrics, prefix=prefix)

    # Benchmark torchcodec on CPU
    cpu_output = temp_dir / "torchcodec_cpu.mp4"
    times, _nvenc_metrics = bench(
        encode_torchcodec,
        frames=frames,
        output_path=str(cpu_output),
        device="cpu",
        average_over=args.average_over,
    )
    report_stats(times, frames.shape[0], prefix="VideoEncoder on CPU")

    # Benchmark FFmpeg CLI on CPU
    ffmpeg_cpu_output = temp_dir / "ffmpeg_cpu.mp4"
    times, _nvenc_metrics = bench(
        encode_ffmpeg_cli,
        frames=frames,
        raw_path=str(raw_frames_path),
        output_path=str(ffmpeg_cpu_output),
        device="cpu",
        skip_write_frames=args.skip_write_frames,
        average_over=args.average_over,
    )
    prefix = "FFmpeg CLI on CPU  "
    report_stats(times, frames.shape[0], prefix=prefix)

    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
