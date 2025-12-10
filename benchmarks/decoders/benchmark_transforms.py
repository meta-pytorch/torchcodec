import tracemalloc
import math
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter_ns

import torch
from torch import Tensor
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import v2


def bench(f, *args, num_exp, warmup=1) -> Tensor:

    for _ in range(warmup):
        f(*args)

    times = []
    for _ in range(num_exp):
        start = perf_counter_ns()
        f(*args)
        end = perf_counter_ns()
        times.append(end - start)
    return torch.tensor(times).float()


def report_stats(times: Tensor, unit: str = "ms", prefix: str = "") -> float:
    mul = {
        "ns": 1,
        "µs": 1e-3,
        "ms": 1e-6,
        "s": 1e-9,
    }[unit]
    times = times * mul
    std = times.std().item()
    med = times.median().item()
    mean = times.mean().item()
    min = times.min().item()
    max = times.max().item()
    print(
        f"{prefix:<45} {med = :.2f}, {mean = :.2f} +- {std:.2f}, {min = :.2f}, {max = :.2f} - in {unit}"
    )


def torchvision_resize(
    path: Path, pts_seconds: list[float], dims: tuple[int, int], num_threads: int
) -> Tensor:
    decoder = VideoDecoder(
        path, seek_mode="approximate", num_ffmpeg_threads=num_threads
    )
    raw_frames = decoder.get_frames_played_at(pts_seconds)
    transformed_frames = v2.Resize(size=dims)(raw_frames.data)
    assert len(transformed_frames) == len(pts_seconds)
    return transformed_frames


def torchvision_crop(
    path: Path, pts_seconds: list[float], dims: tuple[int, int], num_threads: int
) -> Tensor:
    decoder = VideoDecoder(
        path, seek_mode="approximate", num_ffmpeg_threads=num_threads
    )
    raw_frames = decoder.get_frames_played_at(pts_seconds)
    transformed_frames = v2.CenterCrop(size=dims)(raw_frames.data)
    assert len(transformed_frames) == len(pts_seconds)
    return transformed_frames


def decoder_resize(
    path: Path, pts_seconds: list[float], dims: tuple[int, int], num_threads: int
) -> Tensor:
    decoder = VideoDecoder(
        path,
        transforms=[v2.Resize(size=dims)],
        seek_mode="approximate",
        num_ffmpeg_threads=num_threads,
    )
    transformed_frames = decoder.get_frames_played_at(pts_seconds).data
    assert len(transformed_frames) == len(pts_seconds)
    return transformed_frames.data


def decoder_crop(
    path: Path, pts_seconds: list[float], dims: tuple[int, int], num_threads: int
) -> Tensor:
    decoder = VideoDecoder(
        path,
        transforms=[v2.CenterCrop(size=dims)],
        seek_mode="approximate",
        num_ffmpeg_threads=num_threads,
    )
    transformed_frames = decoder.get_frames_played_at(pts_seconds).data
    assert len(transformed_frames) == len(pts_seconds)
    return transformed_frames


def main():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="path to file", required=True)
    parser.add_argument(
        "--num-exp",
        type=int,
        default=5,
        help="number of runs to average over",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="number of threads to use; 0 means FFmpeg decides",
    )
    parser.add_argument(
        "--total-frame-fractions",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.05, 0.1],
    )
    parser.add_argument(
        "--input-dimension-fractions",
        nargs="+",
        type=float,
        default=[0.5, 0.25, 0.125],
    )

    args = parser.parse_args()
    path = Path(args.path)

    metadata = VideoDecoder(path).metadata
    duration = metadata.duration_seconds

    print(
        f"Benchmarking {path.name}, duration: {duration}, codec: {metadata.codec}, averaging over {args.num_exp} runs:"
    )

    input_height = metadata.height
    input_width = metadata.width
    for num_fraction in args.total_frame_fractions:
        num_frames_to_sample = math.ceil(metadata.num_frames * num_fraction)
        print(
            f"Sampling {num_fraction * 100}%, {num_frames_to_sample}, of {metadata.num_frames} frames"
        )
        uniform_timestamps = [
            i * duration / num_frames_to_sample for i in range(num_frames_to_sample)
        ]

        for dims_fraction in args.input_dimension_fractions:
            dims = (int(input_height * dims_fraction), int(input_width * dims_fraction))

            times = bench(
                torchvision_resize,
                path,
                uniform_timestamps,
                dims,
                args.num_threads,
                num_exp=args.num_exp,
            )
            report_stats(times, prefix=f"torchvision_resize({dims})")

            times = bench(
                decoder_resize,
                path,
                uniform_timestamps,
                dims,
                args.num_threads,
                num_exp=args.num_exp,
            )
            report_stats(times, prefix=f"decoder_resize({dims})")

            times = bench(
                torchvision_crop,
                path,
                uniform_timestamps,
                dims,
                args.num_threads,
                num_exp=args.num_exp,
            )
            report_stats(times, prefix=f"torchvision_crop({dims})")

            times = bench(
                decoder_crop,
                path,
                uniform_timestamps,
                dims,
                args.num_threads,
                num_exp=args.num_exp,
            )
            report_stats(times, prefix=f"decoder_crop({dims})")

            print()


if __name__ == "__main__":
    main()
