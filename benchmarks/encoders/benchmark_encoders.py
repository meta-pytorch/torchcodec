#!/usr/bin/env python3
import shutil
import subprocess
import tempfile
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter_ns

import torch
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder

DEFAULT_VIDEO_PATH = "test/resources/nasa_13013.mp4"
# Alternatively, run this command to generate a longer test video:
#   ffmpeg -f lavfi -i testsrc2=duration=600:size=1280x720:rate=30 -c:v libx264 -pix_fmt yuv420p test/resources/testsrc2_10min.mp4


class NVENCMonitor:
    def __init__(self):
        self.nvidia_process = None
        self.metrics = None

    def __enter__(self):
        self.nvidia_process = subprocess.Popen(
            [
                "nvidia-smi",
                "-lms",
                "50",  # check every 50 ms
                "--query-gpu=utilization.encoder,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,  # capture outputs
            stderr=subprocess.DEVNULL,  # ignore errors
            text=True,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.nvidia_process.terminate()
        stdout, _ = self.nvidia_process.communicate()

        samples = []
        for line in stdout.strip().split("\n"):
            if line.strip():
                res = [float(x.strip()) for x in line.split(",")]
                samples.append({"utilization": res[0], "memory_used": res[1]})

        self.metrics = {
            "utilization": [s["utilization"] for s in samples],
            "memory_used": [s["memory_used"] for s in samples],
        }


def bench(f, average_over=50, warmup=2, gpu_monitoring=False, **f_kwargs):
    for _ in range(warmup):
        f(**f_kwargs)

    times = []
    cm = NVENCMonitor if gpu_monitoring else nullcontext
    with cm() as monitor:
        for _ in range(average_over):
            start = perf_counter_ns()
            f(**f_kwargs)
            end = perf_counter_ns()
            times.append(end - start)

    times_tensor = torch.tensor(times).float()
    if gpu_monitoring:
        nvenc_metrics = monitor.metrics
        nvenc_tensor = torch.tensor(nvenc_metrics["utilization"]).float()
        nvenc_memory_used_tensor = torch.tensor(nvenc_metrics["memory_used"]).float()

    return times_tensor, {
        "utilization": nvenc_tensor if gpu_monitoring else None,
        "memory_used": nvenc_memory_used_tensor if gpu_monitoring else None,
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
        mem_used_max = nvenc_metrics["memory_used"].max().item()
        mem_used_median = nvenc_metrics["memory_used"].median().item()
        util_max = nvenc_metrics["utilization"].max().item()
        # For median utilization, only consider non-zero samples as NVENC is often idle
        util_nonzero = nvenc_metrics["utilization"][nvenc_metrics["utilization"] > 0]
        util_median = util_nonzero.median().item() if len(util_nonzero) > 0 else 0.0

        print(
            f"GPU memory used:      median = {mem_used_median:.1f}, max = {mem_used_max:.1f} MiB"
        )
        print(
            f"NVENC utilization:    median = {util_median:.1f}%, max = {util_max:.1f}%"
        )


def encode_torchcodec(frames, output_path, device="cpu"):
    encoder = VideoEncoder(frames=frames, frame_rate=30)
    if device == "cuda":
        encoder.to_file(dest=output_path, codec="h264_nvenc", extra_options={"qp": 0})
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
        default=10,
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

    decoder = VideoDecoder(str(args.path))
    valid_max_frames = min(args.max_frames, len(decoder))
    frames = decoder.get_frames_in_range(start=0, stop=valid_max_frames).data
    gpu_frames = frames.cuda() if cuda_available else None
    print(
        f"Decoded {frames.shape[0]} frames of size {frames.shape[2]}x{frames.shape[3]}"
    )

    temp_dir = Path(tempfile.mkdtemp())
    raw_frames_path = temp_dir / "input_frames.raw"

    # By default, frames will be written inside the benchmark function
    if args.skip_write_frames:
        write_raw_frames(frames, args.max_frames, str(raw_frames_path))

    # Benchmark torchcodec on GPU
    if cuda_available:
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
            skip_write_frames=args.skip_write_frames,
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
        skip_write_frames=args.skip_write_frames,
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
