#!/usr/bin/env python3
import shutil
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter_ns

import psutil
import torch
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder

# GPU monitoring imports (install with: pip install nvidia-ml-py)
try:
    import pynvml

    GPU_MONITORING_AVAILABLE = True
except ImportError:
    print("To enable GPU monitoring, install pynvml with: pip install nvidia-ml-py")
    GPU_MONITORING_AVAILABLE = False

DEFAULT_VIDEO_PATH = "test/resources/nasa_13013.mp4"
# Alternatively, run this command to generate a longer test video:
#   ffmpeg -f lavfi -i testsrc2=duration=600:size=1280x720:rate=30 -c:v libx264 -pix_fmt yuv420p test/resources/testsrc2_10min.mp4
# DEFAULT_VIDEO_PATH = "test/resources/testsrc2_10min.mp4"
DEFAULT_AVERAGE_OVER = 30
DEFAULT_MAX_FRAMES = 300


def gpu_percent():
    if not GPU_MONITORING_AVAILABLE:
        return 0.0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return 0.0


def bench(f, average_over=50, warmup=2, **f_kwargs):
    for _ in range(warmup):
        f(**f_kwargs)

    times = []
    cpu_utils = []
    gpu_utils = []

    for _ in range(average_over):
        psutil.cpu_percent(interval=None)

        start = perf_counter_ns()
        f(**f_kwargs)
        end = perf_counter_ns()

        cpu_util = psutil.cpu_percent(interval=None)
        gpu_util = gpu_percent()

        times.append(end - start)
        cpu_utils.append(cpu_util)
        gpu_utils.append(gpu_util)

    times_tensor = torch.tensor(times).float()
    cpu_tensor = torch.tensor(cpu_utils).float()
    gpu_tensor = torch.tensor(gpu_utils).float()

    return times_tensor, cpu_tensor, gpu_tensor


def report_stats(
    times, num_frames, cpu_utils=None, gpu_utils=None, prefix="", unit="ms"
):
    mul = {
        "ns": 1,
        "µs": 1e-3,
        "ms": 1e-6,
        "s": 1e-9,
    }[unit]
    unit_times = times * mul
    std = unit_times.std().item()
    med = unit_times.median().item()
    mean = unit_times.mean().item()
    min_time = unit_times.min().item()
    max_time = unit_times.max().item()
    print(
        f"\n{prefix}   {med = :.2f}, {mean = :.2f} +- {std:.2f}, {min_time = :.2f}, {max_time = :.2f} - in {unit}"
    )
    if cpu_utils is not None:
        cpu_avg = cpu_utils.mean().item()
        cpu_peak = cpu_utils.max().item()
        print(f"CPU utilization:      avg = {cpu_avg:.1f}%, peak = {cpu_peak:.1f}%")

    if gpu_utils is not None and gpu_utils.numel() > 0:
        gpu_avg = gpu_utils.mean().item()
        gpu_peak = gpu_utils.max().item()
        print(f"GPU utilization:      avg = {gpu_avg:.1f}%, peak = {gpu_peak:.1f}%")


def encode_torchcodec(frames, output_path, device="cpu"):
    encoder = VideoEncoder(frames=frames, frame_rate=30)
    if device == "cuda":
        encoder.to_file(dest=output_path, codec="h264_nvenc", extra_options={"qp": 1})
    else:
        encoder.to_file(dest=output_path, codec="libx264", crf=0)


def write_raw_frames(frames, num_frames, raw_path):
    # Convert NCHW to NHWC for raw video format
    raw_frames = frames.permute(0, 2, 3, 1).contiguous()[:num_frames]
    with open(raw_path, "wb") as f:
        f.write(raw_frames.cpu().numpy().tobytes())


def write_and_encode_ffmpeg_cli(
    frames, num_frames, raw_path, output_path, device="cpu", skip_write_frames=False
):
    # Write frames during benchmarking function by default unless skip_write_frames flag used
    if not skip_write_frames:
        write_raw_frames(frames, num_frames, raw_path)
    height, width = frames.shape[2], frames.shape[3]

    if device == "cuda":
        codec = "h264_nvenc"
        quality_params = ["-qp", "0"]
    else:
        codec = "libx264"
        quality_params = ["-crf", "0"]

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "30",  # frame_rate is 30
        "-i",
        raw_path,
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
    ]
    ffmpeg_cmd.extend(quality_params)
    # By not setting threads, allow FFmpeg to choose.
    # ffmpeg_cmd.extend(["-threads", "1"])
    # try setting threads on VideoEncoder too?
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
        default=DEFAULT_AVERAGE_OVER,
        help="Number of runs to average over",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Maximum number of frames to decode for benchmarking",
    )
    parser.add_argument(
        "--skip-write-frames",
        action="store_true",
        help="Do not write raw frames in FFmpeg CLI benchmarks",
    )

    args = parser.parse_args()

    print(
        f"Benchmarking up to {args.max_frames} frames from {Path(args.path).name} over {args.average_over} runs:"
    )
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("CUDA not available. GPU benchmarks will be skipped.")

    #  Load up to max_frames frames
    decoder = VideoDecoder(str(args.path))
    frames = decoder.get_frames_in_range(
        start=0, stop=min(args.max_frames, len(decoder))
    ).data
    gpu_frames = frames.cuda()
    print(
        f"Loaded {frames.shape[0]} frames of size {frames.shape[2]}x{frames.shape[3]}"
    )

    temp_dir = Path(tempfile.mkdtemp())
    raw_frames_path = temp_dir / "input_frames.raw"

    # By default, frames will be written inside the benchmark function
    if args.skip_write_frames:
        write_raw_frames(frames, args.max_frames, str(raw_frames_path))

    # Benchmark torchcodec on GPU
    if cuda_available:
        gpu_output = temp_dir / "torchcodec_gpu.mp4"
        times, _cpu_utils, gpu_utils = bench(
            encode_torchcodec,
            frames=gpu_frames,
            output_path=str(gpu_output),
            device="cuda",
            average_over=args.average_over,
            warmup=1,
        )
        report_stats(
            times, frames.shape[0], None, gpu_utils, prefix="VideoEncoder on GPU"
        )
    else:
        print("Skipping VideoEncoder GPU benchmark (CUDA not available)")

    # Benchmark FFmpeg CLI on GPU
    if cuda_available:
        ffmpeg_gpu_output = temp_dir / "ffmpeg_gpu.mp4"
        times, _cpu_utils, gpu_utils = bench(
            write_and_encode_ffmpeg_cli,
            frames=gpu_frames,
            num_frames=args.max_frames,
            raw_path=str(raw_frames_path),
            output_path=str(ffmpeg_gpu_output),
            device="cuda",
            skip_write_frames=args.skip_write_frames,
            average_over=args.average_over,
            warmup=1,
        )
        prefix = "FFmpeg CLI on GPU  "
        report_stats(times, frames.shape[0], None, gpu_utils, prefix=prefix)
    else:
        print("Skipping FFmpeg CLI GPU benchmark (CUDA not available)")

    # Benchmark torchcodec on CPU
    cpu_output = temp_dir / "torchcodec_cpu.mp4"
    times, cpu_utils, _gpu_utils = bench(
        encode_torchcodec,
        frames=frames,
        output_path=str(cpu_output),
        device="cpu",
        average_over=args.average_over,
        warmup=1,
    )
    report_stats(times, frames.shape[0], cpu_utils, None, prefix="VideoEncoder on CPU")

    # Benchmark FFmpeg CLI on CPU
    ffmpeg_cpu_output = temp_dir / "ffmpeg_cpu.mp4"
    times, cpu_utils, _gpu_utils = bench(
        write_and_encode_ffmpeg_cli,
        frames=frames,
        num_frames=args.max_frames,
        raw_path=str(raw_frames_path),
        output_path=str(ffmpeg_cpu_output),
        device="cpu",
        skip_write_frames=args.skip_write_frames,
        average_over=args.average_over,
        warmup=1,
    )
    prefix = "FFmpeg CLI on CPU  "
    report_stats(times, frames.shape[0], cpu_utils, None, prefix=prefix)

    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
