import math
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter_ns

import torch
from torch import Tensor
from torchcodec._core import add_video_stream, create_from_file, get_frames_by_pts
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import v2

DEFAULT_NUM_EXP = 20


def bench(f, *args, num_exp=DEFAULT_NUM_EXP, warmup=1) -> Tensor:

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
    path: Path, pts_seconds: list[float], dims: tuple[int, int]
) -> None:
    decoder = create_from_file(str(path))
    add_video_stream(decoder)
    raw_frames, *_ = get_frames_by_pts(decoder, timestamps=pts_seconds)
    return v2.functional.resize(raw_frames, size=dims)


def torchvision_crop(
    path: Path, pts_seconds: list[float], dims: tuple[int, int], x: int, y: int
) -> None:
    decoder = create_from_file(str(path))
    add_video_stream(decoder)
    raw_frames, *_ = get_frames_by_pts(decoder, timestamps=pts_seconds)
    return v2.functional.crop(raw_frames, top=y, left=x, height=dims[0], width=dims[1])


def decoder_native_resize(
    path: Path, pts_seconds: list[float], dims: tuple[int, int]
) -> None:
    decoder = create_from_file(str(path))
    add_video_stream(decoder, transform_specs=f"resize, {dims[0]}, {dims[1]}")
    return get_frames_by_pts(decoder, timestamps=pts_seconds)[0]


def decoder_native_crop(
    path: Path, pts_seconds: list[float], dims: tuple[int, int], x: int, y: int
) -> None:
    decoder = create_from_file(str(path))
    add_video_stream(decoder, transform_specs=f"crop, {dims[0]}, {dims[1]}, {x}, {y}")
    return get_frames_by_pts(decoder, timestamps=pts_seconds)[0]


def main():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="path to file", required=True)
    parser.add_argument(
        "--num-exp",
        type=int,
        default=DEFAULT_NUM_EXP,
        help="number of runs to average over",
    )

    args = parser.parse_args()
    path = Path(args.path)

    metadata = VideoDecoder(path).metadata
    duration = metadata.duration_seconds

    print(
        f"Benchmarking {path.name}, duration: {duration}, codec: {metadata.codec}, averaging over {args.num_exp} runs:"
    )

    height = metadata.height
    width = metadata.width
    dimensions = [
        (int(height * 0.5), int(width * 0.5)),
        (int(height * 0.25), int(width * 0.25)),
        (int(height * 0.125), int(width * 0.125)),
    ]
    fraction_of_total_frames_to_sample = [0.01, 0.05, 0.1]

    for fraction in fraction_of_total_frames_to_sample:
        print(f"Sampling {fraction * 100}% of {metadata.num_frames} frames")
        num_frames_to_sample = math.ceil(metadata.num_frames * fraction)
        uniform_timestamps = [
            i * duration / num_frames_to_sample for i in range(num_frames_to_sample)
        ]

        for dims in dimensions:
            times = bench(
                torchvision_resize, path, uniform_timestamps, dims, num_exp=args.num_exp
            )
            report_stats(times, prefix=f"torchvision_resize({dims})")

            times = bench(
                decoder_native_resize,
                path,
                uniform_timestamps,
                dims,
                num_exp=args.num_exp,
            )
            report_stats(times, prefix=f"decoder_native_resize({dims})")
            print()

            center_x = (height - dims[0]) // 2
            center_y = (width - dims[1]) // 2
            times = bench(
                torchvision_crop,
                path,
                uniform_timestamps,
                dims,
                center_x,
                center_y,
                num_exp=args.num_exp,
            )
            report_stats(
                times, prefix=f"torchvision_crop({dims}, {center_x}, {center_y})"
            )

            times = bench(
                decoder_native_crop,
                path,
                uniform_timestamps,
                dims,
                center_x,
                center_y,
                num_exp=args.num_exp,
            )
            report_stats(
                times, prefix=f"decoder_native_crop({dims}, {center_x}, {center_y})"
            )
            print()


if __name__ == "__main__":
    main()
