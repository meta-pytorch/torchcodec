"""Benchmark torchcodec's JPEG and PNG image encoders against torchvision and PIL.

Encodes a single 720p RGB image, comparing three destinations:
- "file"     : write to a file path on disk
- "file_like": write to an in-memory file-like (io.BytesIO)
- "tensor"   : get the encoded bytes back as a 1-D uint8 tensor

For the "tensor" destination, torchcodec and torchvision both return the encoded
bytes as a tensor natively (torchcodec via encode_*(dest=None)); on CUDA both
produce a device tensor which we .cpu() (you'd need the bytes on the host to
write them anywhere). PIL has no such API, so it encodes into a BytesIO and wraps
the buffer with torch.frombuffer(buf.getbuffer()) (getbuffer() avoids the copy
that getvalue() makes). The PIL path includes the CHW-tensor -> numpy conversion
in the timed region. On CUDA, PIL is skipped (CPU-only) and torchvision has no
file/file-like encode API. PNG is CPU-only (no CUDA PNG encoder). Run:

    python benchmarks/encoders/benchmark_image_encoders.py --num-exp 50

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
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from PIL import Image
from torch import Tensor

torch.set_num_threads(1)

from torchcodec.encoders._image_encoders import (  # noqa: E402
    encode_jpeg as tc_encode_jpeg,
    encode_png as tc_encode_png,
)
from torchvision.io import (  # noqa: E402
    encode_jpeg as tv_encode_jpeg,
    encode_png as tv_encode_png,
    write_file as tv_write_file,
    write_jpeg as tv_write_jpeg,
    write_png as tv_write_png,
)

DEFAULT_NUM_EXP = 50
RESOLUTION = (1280, 720)  # (W, H)
DESTINATIONS = ("file", "file_like", "tensor")
FORMATS = ("jpeg", "png")


def bench(f, *, num_exp, warmup=5) -> Tensor:
    for _ in range(warmup):
        f()
    times = []
    for _ in range(num_exp):
        start = perf_counter_ns()
        f()
        end = perf_counter_ns()
        times.append(end - start)
    return torch.tensor(times).float()


def _stats_ms(times: Tensor) -> tuple[float, float]:
    ms = times * 1e-6
    return ms.median().item(), ms.std().item()


def cuda_synced(fn):
    # Wrap fn so device-side work completes inside the timed region.
    def g():
        fn()
        torch.cuda.synchronize()

    return g


def load_source_image(override: Path | None) -> Tensor:
    """Return a (3, 720, 1280) uint8 RGB CHW tensor."""
    if override is not None:
        img = Image.open(override).convert("RGB")
    else:
        try:
            import scipy.datasets

            img = Image.fromarray(scipy.datasets.face(), mode="RGB")
        except Exception as e:
            print(f"scipy.datasets.face() unavailable ({e}); using synthetic image.")
            rng = np.random.RandomState(0)
            h, w = 768, 1024
            base = np.linspace(0, 255, w, dtype=np.float32)[None, :].repeat(h, 0)
            chan = np.clip(base + rng.randn(h, w) * 40, 0, 255).astype(np.uint8)
            img = Image.fromarray(
                np.stack([chan, np.roll(chan, 30, 1), np.roll(chan, 60, 0)], -1), "RGB"
            )
    img = img.resize(RESOLUTION, Image.LANCZOS)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()


def make_encode_fn(backend, fmt, dest, img, path, param):
    """Return a zero-arg callable that encodes `img` (in `fmt`) to `dest`, or
    None if the backend doesn't support that destination. `param` is the quality
    (jpeg) or compression_level (png). torchcodec/torchvision take the encode
    param positionally so this stays format-agnostic."""
    is_png = fmt == "png"
    if backend == "torchcodec":
        tc_encode = tc_encode_png if is_png else tc_encode_jpeg
        if dest == "file":
            return lambda: tc_encode(img, path, param)
        if dest == "file_like":
            return lambda: tc_encode(img, io.BytesIO(), param)
        # tensor: native dest=None. On CUDA it returns a device tensor
        # (zero-copy); .cpu() brings it to host to match torchvision's path.
        return lambda: tc_encode(img, None, param).cpu()

    if backend == "torchvision":
        if is_png:
            if dest == "file":
                return lambda: tv_write_png(img, str(path), param)
            if dest == "tensor":
                return lambda: tv_encode_png(img, param).cpu()
            return None  # no file-like encode API
        if dest == "file":
            if img.is_cuda:
                # torchvision has no CUDA file API, so we do what a user would:
                # encode -> bring bytes to host -> write_file.
                return lambda: tv_write_file(
                    str(path), tv_encode_jpeg(img, quality=param).cpu()
                )
            return lambda: tv_write_jpeg(img, str(path), quality=param)
        if dest == "tensor":
            # encode_jpeg returns the bytes on the input tensor's device, so on
            # CUDA we .cpu() them: that's what you'd have to do to actually write
            # them out, and it makes the comparison fair with torchcodec, which
            # already returns host bytes. On CPU .cpu() is a no-op.
            return lambda: tv_encode_jpeg(img, quality=param).cpu()
        return None  # no file-like encode API

    if backend == "PIL":
        if img.is_cuda:
            return None  # PIL is CPU-only

        save_format = "PNG" if is_png else "JPEG"
        save_kwargs = {"compress_level": param} if is_png else {"quality": param}

        def f():
            # Convert the CHW tensor to an HWC numpy image, as part of the timing.
            pil = Image.fromarray(img.permute(1, 2, 0).numpy())
            if dest == "file":
                pil.save(path, format=save_format, **save_kwargs)
            elif dest == "file_like":
                pil.save(io.BytesIO(), format=save_format, **save_kwargs)
            else:  # tensor
                buf = io.BytesIO()
                pil.save(buf, format=save_format, **save_kwargs)
                return torch.frombuffer(buf.getbuffer(), dtype=torch.uint8)

        return f

    return None


def make_batch_encode_fn(backend, imgs, quality):
    """Encode a batch of images to host byte tensors. torchcodec has no batch
    encode API, so we loop; torchvision has a batched encode_jpeg(list). Both
    return the encoded bytes on the host (torchvision's live on the input
    device, so we .cpu() them)."""
    if backend == "torchcodec":
        return lambda: [tc_encode_jpeg(img, quality=quality).cpu() for img in imgs]

    if backend == "torchvision":
        return lambda: [t.cpu() for t in tv_encode_jpeg(list(imgs), quality=quality)]

    return None


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--num-exp", type=int, default=DEFAULT_NUM_EXP)
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality")
    parser.add_argument(
        "--compression-level", type=int, default=6, help="PNG compression level"
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--image", type=Path, default=None, help="override source image"
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 16, 64],
        help="batch sizes for the CUDA batch-encode comparison",
    )
    args = parser.parse_args()

    print(
        f"torch.get_num_threads()={torch.get_num_threads()}, 720p, "
        f"quality={args.quality}, compression_level={args.compression_level}"
    )

    source = load_source_image(args.image)
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "out"

    backends = ["torchcodec", "torchvision", "PIL"]
    rows = []  # (fmt, device, dest, {backend: (median, std)})
    for fmt in FORMATS:
        param = args.compression_level if fmt == "png" else args.quality
        for device in args.devices:
            if fmt == "png" and device == "cuda":
                continue  # no CUDA PNG encoder
            img = source.to(device)
            for dest in DESTINATIONS:
                results = {}
                for backend in backends:
                    fn = make_encode_fn(backend, fmt, dest, img, path, param)
                    if fn is None:
                        results[backend] = None
                        continue
                    if device == "cuda":
                        fn = cuda_synced(fn)
                    try:
                        fn()  # smoke
                        results[backend] = _stats_ms(bench(fn, num_exp=args.num_exp))
                    except Exception as e:
                        results[backend] = None
                        print(
                            f"  [skip] {fmt} {device} {dest} {backend}: {str(e)[:80]}"
                        )
                rows.append((fmt, device, dest, results))

    print(
        f"\n## Encode 720p image latency (ms), median +- std, 1 thread, "
        f"{args.num_exp} runs\n"
    )
    hdr = (
        f"{'fmt':<6}{'device':<7}{'dest':<11}{'torchcodec':>17}{'tv':>17}"
        f"{'PIL':>17}{'tv/tc':>9}{'PIL/tc':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    def fmt_ms(v):
        return f"{v[0]:.2f} +- {v[1]:.1f}" if v is not None else "N/A"

    def ratio(num, den):
        if num is None or den is None or den[0] == 0:
            return "N/A"
        return f"{num[0] / den[0]:.2f}x"

    for fmt, device, dest, r in rows:
        tc, tv, pil = r["torchcodec"], r["torchvision"], r["PIL"]
        print(
            f"{fmt:<6}{device:<7}{dest:<11}{fmt_ms(tc):>17}{fmt_ms(tv):>17}"
            f"{fmt_ms(pil):>17}{ratio(tv, tc):>9}{ratio(pil, tc):>9}"
        )

    print("\n(tv/tc and PIL/tc > 1.0 mean torchcodec is faster.)")

    # ---- CUDA batch encode: torchcodec loop vs torchvision batch API ----
    if "cuda" in args.devices:
        print(
            f"\n## Batch encode 720p JPEG on CUDA -> host byte tensors, "
            f"median +- std ms, {args.num_exp} runs\n"
        )
        bhdr = (
            f"{'batch':<7}{'torchcodec':>17}{'tv':>17}{'tv/tc':>9}"
            f"{'tc/img':>17}{'tv/img':>17}"
        )
        print(bhdr)
        print("-" * len(bhdr))
        for bs in args.batch_sizes:
            imgs = [source.to("cuda") for _ in range(bs)]
            b = {}
            for backend in ("torchcodec", "torchvision"):
                fn = cuda_synced(make_batch_encode_fn(backend, imgs, args.quality))
                fn()  # smoke
                b[backend] = _stats_ms(bench(fn, num_exp=args.num_exp))
            tc, tv = b["torchcodec"], b["torchvision"]
            tc_img = (tc[0] / bs, tc[1] / bs)
            tv_img = (tv[0] / bs, tv[1] / bs)
            print(
                f"{bs:<7}{fmt_ms(tc):>17}{fmt_ms(tv):>17}{ratio(tv, tc):>9}"
                f"{fmt_ms(tc_img):>17}{fmt_ms(tv_img):>17}"
            )
        print("\n(tv/tc > 1.0 means torchcodec's loop is faster than tv's batch API.)")

    tmp.cleanup()


if __name__ == "__main__":
    main()
