#!/usr/bin/env python3
import shutil
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter_ns

import torch
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder

DEFAULT_VIDEO_PATH = "test/resources/nasa_13013.mp4"
# Alternatively, run this command to generate a longer test video:
#   ffmpeg -f lavfi -i testsrc2=duration=600:size=1280x720:rate=30 -c:v libx264 -pix_fmt yuv420p test/resources/testsrc2_10min.mp4
# DEFAULT_VIDEO_PATH = "test/resources/testsrc2_10min.mp4"
DEFAULT_AVERAGE_OVER = 30
DEFAULT_MAX_FRAMES = 300


def monitor_nvenc_during_encoding(encoding_func, **kwargs):
    nvidia_process = subprocess.Popen(
        [
            "nvidia-smi",
            "-lms",
            "50",
            "--query-gpu=utilization.encoder,memory.used",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    try:
        encoding_func(**kwargs)
    finally:
        nvidia_process.terminate()
        try:
            stdout, _ = nvidia_process.communicate()
        except subprocess.TimeoutExpired:
            nvidia_process.kill()
            stdout, _ = nvidia_process.communicate()

    nvidia_samples = []
    for line in stdout.strip().split("\n"):
        if line.strip():
            values = [float(x.strip()) for x in line.split(",")]
            nvidia_samples.append({"utilization": values[0], "memory_used": values[1]})

    max_util = max((s["utilization"] for s in nvidia_samples), default=0.0)
    max_memory = max((s["memory_used"] for s in nvidia_samples), default=0.0)

    return {"utilization": max_util, "memory_used": max_memory}


def bench(f, average_over=50, warmup=2, **f_kwargs):
    for _ in range(warmup):
        f(**f_kwargs)

    times = []
    nvenc_utils = []
    nvenc_memory_used = []

    for _ in range(average_over):
        start = perf_counter_ns()
        nvenc_metrics = monitor_nvenc_during_encoding(f, **f_kwargs)
        end = perf_counter_ns()

        times.append(end - start)
        nvenc_utils.append(nvenc_metrics["utilization"])
        nvenc_memory_used.append(nvenc_metrics["memory_used"])

    times_tensor = torch.tensor(times).float()
    nvenc_tensor = torch.tensor(nvenc_utils).float()
    nvenc_memory_used_tensor = torch.tensor(nvenc_memory_used).float()

    return times_tensor, {
        "utilization": nvenc_tensor,
        "memory_used": nvenc_memory_used_tensor,
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
    std = unit_times.std().item()
    med = unit_times.median().item()
    mean = unit_times.mean().item()
    min = unit_times.min().item()
    max = unit_times.max().item()
    print(
        f"\n{prefix}   {med = :.2f}, {mean = :.2f} +- {std:.2f}, {min = :.2f}, {max = :.2f} - in {unit}, fps = {fps:.1f}"
    )

    if nvenc_metrics is not None:
        # NVENC metrics structure - show median and peak values
        util_median = nvenc_metrics["utilization"].median().item()
        util_peak = nvenc_metrics["utilization"].max().item()
        mem_used_median = nvenc_metrics["memory_used"].median().item()
        mem_used_peak = nvenc_metrics["memory_used"].max().item()

        print(
            f"NVENC utilization:    median = {util_median:.1f}%, peak = {util_peak:.1f}%"
        )
        print(
            f"GPU memory used:      median = {mem_used_median:.1f}, peak = {mem_used_peak:.1f} MiB"
        )


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
    frames, num_frames, raw_path, output_path, device="cpu", write_frames=False
):
    # Rewrite frames during benchmarking function if write_frames flag used
    if write_frames:
        write_raw_frames(frames, num_frames, raw_path)

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
        "30",  # frame_rate is 30
        "-i",
        raw_path,
        "-c:v",
        "h264_nvenc" if device == "cuda" else "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    # quality_params = ["-qp", "0"] if device == "cuda" else ["-crf", "0"]
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
        "--write-frames",
        action="store_true",
        help="Include raw frame writing time in FFmpeg CLI benchmarks for fairer comparison with tensor-based workflows",
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
    valid_max_frames = min(args.max_frames, len(decoder))
    frames = decoder.get_frames_in_range(start=0, stop=valid_max_frames).data
    gpu_frames = frames.cuda()
    print(
        f"Decoded {frames.shape[0]} frames of size {frames.shape[2]}x{frames.shape[3]}"
    )

    temp_dir = Path(tempfile.mkdtemp())
    raw_frames_path = temp_dir / "input_frames.raw"

    # Write frames once outside benchmarking when --write-frames is False
    # When --write-frames is True, frames will be written inside the benchmark function
    if not args.write_frames:
        write_raw_frames(frames, valid_max_frames, str(raw_frames_path))

    # Benchmark torchcodec on GPU
    if cuda_available:
        gpu_output = temp_dir / "torchcodec_gpu.mp4"
        times, nvenc_metrics = bench(
            encode_torchcodec,
            frames=gpu_frames,
            output_path=str(gpu_output),
            device="cuda",
            average_over=args.average_over,
        )
        report_stats(
            times, frames.shape[0], nvenc_metrics, prefix="VideoEncoder on GPU"
        )
    else:
        print("Skipping VideoEncoder GPU benchmark (CUDA not available)")

    # Benchmark FFmpeg CLI on GPU
    if cuda_available:
        ffmpeg_gpu_output = temp_dir / "ffmpeg_gpu.mp4"
        times, nvenc_metrics = bench(
            write_and_encode_ffmpeg_cli,
            frames=gpu_frames,
            num_frames=valid_max_frames,
            raw_path=str(raw_frames_path),
            output_path=str(ffmpeg_gpu_output),
            device="cuda",
            write_frames=args.write_frames,
            average_over=args.average_over,
        )
        prefix = "FFmpeg CLI on GPU  "
        report_stats(times, frames.shape[0], nvenc_metrics, prefix=prefix)
    else:
        print("Skipping FFmpeg CLI GPU benchmark (CUDA not available)")

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
        write_and_encode_ffmpeg_cli,
        frames=frames,
        num_frames=valid_max_frames,
        raw_path=str(raw_frames_path),
        output_path=str(ffmpeg_cpu_output),
        device="cpu",
        write_frames=args.write_frames,
        average_over=args.average_over,
    )
    prefix = "FFmpeg CLI on CPU  "
    report_stats(times, frames.shape[0], prefix=prefix)

    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
