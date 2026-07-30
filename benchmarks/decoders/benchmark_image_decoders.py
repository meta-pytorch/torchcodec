"""Benchmark torchcodec's dedicated image decoders against PIL, torchvision, and
torchcodec's FFmpeg video-decoder path, across formats and resolutions.

All backends are pinned to a single thread and decode to a comparable
(C, H, W) uint8 RGB tensor. Run:

    python benchmarks/decoders/benchmark_image_decoders.py --num-exp 50

See --help for options.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import io
import tempfile
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from PIL import Image
from torch import Tensor

torch.set_num_threads(1)

try:
    from PIL import AvifImagePlugin

    AvifImagePlugin.DEFAULT_MAX_THREADS = 1
except ImportError:
    pass

from torchcodec._core.ops import (  # noqa: E402
    add_video_stream,
    create_from_file,
    create_from_tensor,
    get_next_frame,
)
from torchcodec.decoders._image_decoders import (  # noqa: E402
    decode_avif,
    decode_gif,
    decode_heic,
    decode_jpeg,
    decode_png,
    decode_webp,
)

from torchvision.io import (  # noqa: E402
    decode_gif as tv_decode_gif,
    decode_jpeg as tv_decode_jpeg,
    decode_png as tv_decode_png,
    decode_webp as tv_decode_webp,
    read_file as tv_read_file,
)
from torchvision.transforms.v2.functional import pil_to_tensor  # noqa: E402

DEFAULT_NUM_EXP = 50

RESOLUTIONS = {
    "320p": (568, 320),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}

_TC_DECODERS = {
    "jpeg": decode_jpeg,
    "png": decode_png,
    "webp": decode_webp,
    "gif": decode_gif,
    "avif": decode_avif,
    "heic": decode_heic,
}

# torchvision has no avif/heic decoder, they're in torchvision-extra-decoders,
# but we don't bother (they're broken anyway). decode_gif is RGB-only and takes
# no mode argument.
_TV_DECODERS = {
    "jpeg": lambda d: tv_decode_jpeg(d, "RGB"),
    "png": lambda d: tv_decode_png(d, "RGB"),
    "webp": lambda d: tv_decode_webp(d, "RGB"),
    "gif": lambda d: tv_decode_gif(d),
}

_EXT = {
    "jpeg": "jpg",
    "png": "png",
    "webp": "webp",
    "gif": "gif",
    "avif": "avif",
    "heic": "heic",
}


def bench(f, *args, num_exp=DEFAULT_NUM_EXP, warmup=5, **kwargs) -> Tensor:
    for _ in range(warmup):
        f(*args, **kwargs)
    times = []
    for _ in range(num_exp):
        start = perf_counter_ns()
        f(*args, **kwargs)
        end = perf_counter_ns()
        times.append(end - start)
    return torch.tensor(times).float()


def _median_ms(times: Tensor) -> float:
    return (times * 1e-6).median().item()


# ---------------------------------------------------------------------------
# Test image generation
# ---------------------------------------------------------------------------
def load_source_image(override: Path | None) -> Image.Image:
    if override is not None:
        return Image.open(override).convert("RGB")
    try:
        import scipy.datasets

        arr = scipy.datasets.face()  # (768, 1024, 3) real photo, cached
        return Image.fromarray(arr, mode="RGB")
    except Exception as e:
        print(
            f"scipy.datasets.face() unavailable ({e}); using textured synthetic image."
        )
        rng = np.random.RandomState(0)
        h, w = 768, 1024
        base = np.linspace(0, 255, w, dtype=np.float32)[None, :].repeat(h, 0)
        noise = rng.randn(h, w) * 40
        chan = np.clip(base + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(
            np.stack([chan, np.roll(chan, 30, 1), np.roll(chan, 60, 0)], -1), "RGB"
        )


def encode_images(image: Image.Image, resolutions, out_dir: Path) -> dict:
    """Encode the same image at each resolution into every format. Returns
    {(fmt, res_label): path}."""
    import pillow_heif

    pillow_heif.register_heif_opener()  # needed to save HEIC via PIL

    paths = {}
    for res_label in resolutions:
        w, h = RESOLUTIONS[res_label]
        resized = image.resize((w, h), Image.LANCZOS)
        for fmt in _EXT:
            path = out_dir / f"img_{res_label}.{_EXT[fmt]}"
            if fmt == "jpeg":
                resized.save(path, format="JPEG", quality=90)
            elif fmt == "png":
                resized.save(path, format="PNG")
            elif fmt == "webp":
                resized.save(path, format="WEBP", quality=90)
            elif fmt == "gif":
                resized.convert("P", palette=Image.ADAPTIVE).save(path, format="GIF")
            elif fmt == "avif":
                resized.save(path, format="AVIF", quality=90)
            elif fmt == "heic":
                resized.save(path, format="HEIF", quality=90)
            paths[(fmt, res_label)] = path
    return paths


# ---------------------------------------------------------------------------
# Backends: decode to a (C, H, W) uint8 RGB tensor, either from a preloaded
# uint8 byte tensor (from_file=False, timing pure decode) or straight from a
# file path (from_file=True, timing decode + I/O).
# ---------------------------------------------------------------------------
def decode_torchcodec(fmt: str, path: Path, data: Tensor, from_file: bool) -> Tensor:
    return _TC_DECODERS[fmt](path if from_file else data, mode="RGB")


def decode_torchvision(fmt: str, path: Path, data: Tensor, from_file: bool) -> Tensor:
    return _TV_DECODERS[fmt](tv_read_file(str(path)) if from_file else data)


def decode_pil(path: Path, raw: bytes, from_file: bool) -> Tensor:
    img = Image.open(path if from_file else io.BytesIO(raw)).convert("RGB")
    return pil_to_tensor(img)


def decode_ffmpeg(path: Path, data: Tensor, from_file: bool) -> Tensor:
    if from_file:
        dec = create_from_file(str(path), "approximate")
    else:
        dec = create_from_tensor(data, "approximate")
    add_video_stream(dec, num_threads=1)
    frame, _, _ = get_next_frame(dec)
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions", nargs="+", default=list(RESOLUTIONS), choices=list(RESOLUTIONS)
    )
    parser.add_argument("--num-exp", type=int, default=DEFAULT_NUM_EXP)
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="override source image (default: scipy.datasets.face())",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--from-bytes",
        dest="from_file",
        action="store_false",
        default=False,
        help="decode from preloaded in-memory bytes; times pure decode (default)",
    )
    source.add_argument(
        "--from-file",
        dest="from_file",
        action="store_true",
        help="decode straight from a file path; times decode + I/O",
    )
    args = parser.parse_args()

    print(
        f"torch.get_num_threads()={torch.get_num_threads()}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}"
    )
    print(
        "Note: libavif/dav1d internal threading is not exposed by decode_avif (library default)."
    )
    print(f"Source: {'file path' if args.from_file else 'in-memory bytes'}")

    image = load_source_image(args.image)
    tmp = tempfile.TemporaryDirectory()
    paths = encode_images(image, args.resolutions, Path(tmp.name))

    backends = ["torchcodec", "PIL", "torchvision", "FFmpeg"]
    rows = []  # (fmt, res, sizes_kb, {backend: median_ms})

    for fmt in _EXT:
        for res_label in args.resolutions:
            path = paths[(fmt, res_label)]
            raw = path.read_bytes()
            data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            size_kb = len(raw) / 1024.0
            results = {}
            for backend in backends:
                # torchvision has no avif/heic decoder; leave those cells blank.
                if backend == "torchvision" and fmt not in _TV_DECODERS:
                    results[backend] = None
                    continue
                try:
                    ff = args.from_file
                    if backend == "torchcodec":
                        fn = partial(decode_torchcodec, fmt, path, data, ff)
                    elif backend == "PIL":
                        fn = partial(decode_pil, path, raw, ff)
                    elif backend == "torchvision":
                        fn = partial(decode_torchvision, fmt, path, data, ff)
                    else:
                        fn = partial(decode_ffmpeg, path, data, ff)
                    fn()  # smoke
                    times = bench(fn, num_exp=args.num_exp)
                    results[backend] = _median_ms(times)
                except Exception as e:
                    results[backend] = None
                    print(f"  [skip] {fmt} {res_label} {backend}: {str(e)[:80]}")
            rows.append((fmt, res_label, size_kb, results))

    # ---- results table ----
    print(f"\n## Decode-to-tensor median latency (ms), 1 thread, {args.num_exp} runs\n")
    hdr = (
        f"{'format':<7}{'res':<7}{'size KB':>9}{'torchcodec':>12}{'PIL':>9}"
        f"{'tv':>9}{'FFmpeg':>9}{'PIL/tc':>9}{'tv/tc':>9}{'FFmpeg/tc':>11}"
    )
    print(hdr)
    print("-" * len(hdr))

    def fmt_ms(v):
        return f"{v:.3f}" if v is not None else "N/A"

    def ratio(num, den):
        if num is None or den is None or den == 0:
            return "N/A"
        return f"{num / den:.2f}x"

    for fmt, res_label, size_kb, r in rows:
        tc, pil, tv, ff = (
            r["torchcodec"],
            r["PIL"],
            r["torchvision"],
            r["FFmpeg"],
        )
        print(
            f"{fmt:<7}{res_label:<7}{size_kb:>9.1f}{fmt_ms(tc):>12}{fmt_ms(pil):>9}"
            f"{fmt_ms(tv):>9}{fmt_ms(ff):>9}{ratio(pil, tc):>9}{ratio(tv, tc):>9}"
            f"{ratio(ff, tc):>11}"
        )

    print("\n(PIL/tc, tv/tc and FFmpeg/tc > 1.0 mean torchcodec is faster.)")
    tmp.cleanup()


if __name__ == "__main__":
    main()
