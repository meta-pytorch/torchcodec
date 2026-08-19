# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
===============================================
Blocks: build your own decoding pipeline
===============================================

.. warning::

   **The Blocks APIs are under active construction.** They are private
   and unreleased. Signatures and semantics may change without notice. This
   tutorial only exists to show what they will eventually make possible.

:class:`~torchcodec.decoders.VideoDecoder` is a single box that does demuxing,
decoding and color conversion for you. The Blocks APIs expose those three
stages separately:

.. code-block::

   Demuxer  ->  PacketDecoder  ->  ColorConverter
    Packet       DecodedFrame        RGB Frame

The blocks are passive: they never create threads, and they release the GIL.
You decide how they are composed, on which threads, and where to stop. Below
we illustrate a few things this enables: overlapping stages on multiple
threads, accessing raw (YUV) frames, and decoding streams of unknown -
possibly infinite - length.
"""

# %%
# Boilerplate: a test video, and the device we'll run on.
import subprocess
import tempfile
from pathlib import Path

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device = }")

temp_dir = Path(tempfile.mkdtemp())
video_path = temp_dir / "video.mp4"
subprocess.run(
    [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        str(video_path),
    ],
    check=True,
)

# %%
# The three blocks
# ----------------
#
# A pipeline is just a loop. The decoder may need more than one packet before
# it can output a frame, and it buffers a few frames that ``flush()`` returns
# at the end.
#
# ``PacketDecoder`` and ``ColorConverter`` both accept ``device="cuda"``:
# decoding then runs on NVDEC and the color conversion on the GPU, and the
# frames never leave the device. Demuxing always happens on the CPU.
from torchcodec.decoders._blocks import ColorConverter, Demuxer, PacketDecoder

demuxer = Demuxer(video_path)
packet_decoder = PacketDecoder(demuxer, device=device)
color_converter = ColorConverter(device=device)

frames = []
for packet in demuxer:
    for decoded_frame in packet_decoder.decode(packet):
        frames.append(color_converter.convert(decoded_frame))
for decoded_frame in packet_decoder.flush():
    frames.append(color_converter.convert(decoded_frame))

print(f"{len(frames)} frames, {frames[0].data.shape = }, "
      f"{frames[0].pts_seconds = }, {frames[0].data.device = }")

# %%
# Threading: overlapping the stages
# ---------------------------------
#
# Each stage is a generator, so a pipeline is a chain of generators. Inserting
# ``prefetch()`` between two of them puts everything upstream on its own
# thread: the stages then run concurrently, and since the blocks release the
# GIL, that's real parallelism.
import queue
import threading


def demux(demuxer):
    yield from demuxer


def decode(packet_decoder, packets):
    for packet in packets:
        yield from packet_decoder.decode(packet)
    yield from packet_decoder.flush()


def color_convert(color_converter, decoded_frames):
    for decoded_frame in decoded_frames:
        yield color_converter.convert(decoded_frame)


def prefetch(upstream, buffer_size=8):
    # Run `upstream` on a background thread, yielding its items through a
    # bounded queue. The queue applies backpressure: the worker blocks in
    # put() when the buffer is full, so it stays at most buffer_size ahead.
    q = queue.Queue(maxsize=buffer_size)
    eof = object()

    def worker():
        for item in upstream:
            q.put(item)
        q.put(eof)

    threading.Thread(target=worker, daemon=True).start()

    def drain():
        while (item := q.get()) is not eof:
            yield item

    return drain()


def sequential():
    # demux -> decode -> color-convert, all on the calling thread.
    demuxer = Demuxer(video_path)
    packet_decoder = PacketDecoder(demuxer, device=device)
    color_converter = ColorConverter(device=device)
    return color_convert(color_converter, decode(packet_decoder, demux(demuxer)))


def convert_on_own_thread():
    # [demux + decode] on one thread || [color-convert] on another.
    demuxer = Demuxer(video_path)
    packet_decoder = PacketDecoder(demuxer, device=device)
    color_converter = ColorConverter(device=device)
    decoded_frames = prefetch(decode(packet_decoder, demux(demuxer)))
    return color_convert(color_converter, decoded_frames)


def demux_on_own_thread():
    # [demux] on one thread || [decode + color-convert] on another. This is the
    # natural split on CUDA: demuxing is CPU and I/O work, while decoding and
    # color conversion both happen on the GPU, so they belong together.
    demuxer = Demuxer(video_path)
    packet_decoder = PacketDecoder(demuxer, device=device)
    color_converter = ColorConverter(device=device)
    packets = prefetch(demux(demuxer))
    return color_convert(color_converter, decode(packet_decoder, packets))


for pipeline in (sequential, convert_on_own_thread, demux_on_own_thread):
    frames = list(pipeline())
    print(f"{pipeline.__name__}: {len(frames)} frames on {frames[0].data.device}")

# %%
# Where you insert the thread boundaries is up to you, and so is everything
# else: nothing stops you from running one pipeline per file, decoding on the
# CPU while color-converting on the GPU, or feeding frames into your own
# pre-fetching data loader.

# %%
# Raw frames
# ----------
#
# Color conversion is optional. A ``DecodedFrame`` can hand out the decoder's
# own planes as tensor views, with no copy and no conversion.
demuxer = Demuxer(video_path)
packet_decoder = PacketDecoder(demuxer, device=device)
decoded_frame = next(decode(packet_decoder, demux(demuxer)))

raw_frame = decoded_frame.materialize()
Y, U, V = raw_frame.planes
print(f"{raw_frame.pix_fmt = }, {raw_frame.bit_depth = }, "
      f"{raw_frame.colorspace = }, {raw_frame.color_range = }")
print(f"{Y.shape = }, {U.shape = }, {Y.dtype = }, {Y.stride() = }")

# %%
# These are views into the frame's memory: the row stride is the decoder's own
# line size, and the chroma planes of an NVDEC surface are two interleaved
# views over a single plane. Writing through them is visible downstream.
#
# So we can do the color conversion ourselves. Here it's plain PyTorch ops -
# it could just as well be a Triton or CUDA kernel, fused with whatever your
# model needs next.
assert raw_frame.pix_fmt in ("yuv420p", "nv12")  # 8-bit 4:2:0, on CPU and CUDA
assert raw_frame.colorspace == "bt709" and raw_frame.color_range == "tv"


def yuv420_to_rgb(Y, U, V):
    # BT.709, limited range. Chroma is upsampled by nearest neighbour.
    height, width = Y.shape

    def upsample(plane):
        plane = (plane.float() - 128) * (255 / 224)
        return plane.repeat_interleave(2, 0).repeat_interleave(2, 1)[:height, :width]

    y = (Y.float() - 16) * (255 / 219)
    u, v = upsample(U), upsample(V)
    rgb = torch.stack(
        [
            y + 1.5748 * v,
            y - 0.1873 * u - 0.4681 * v,
            y + 1.8556 * u,
        ]
    )
    return rgb.round_().clamp_(0, 255).to(torch.uint8)


ours = yuv420_to_rgb(Y, U, V)
reference = ColorConverter(device=device).convert(decoded_frame).data
print(f"{ours.shape = }, mean abs diff vs ColorConverter: "
      f"{(ours.float() - reference.float()).abs().mean():.2f}")

# %%
# Raw HDR frames
# ~~~~~~~~~~~~~~
#
# Raw planes come at the source's own precision, so a 10-bit HDR video gives
# ``uint16`` planes with all 10 bits intact - no clipping to 8 bits, and no
# tone mapping.
hdr_video_path = temp_dir / "hdr.mp4"
subprocess.run(
    [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=1",
        "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-preset", "ultrafast",
        "-x265-params",
        "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited",
        str(hdr_video_path),
    ],
    check=True,
    capture_output=True,  # x265 logs its banner to stderr no matter what
)

hdr_demuxer = Demuxer(hdr_video_path)
hdr_packet_decoder = PacketDecoder(hdr_demuxer, device=device)
hdr_frame = next(decode(hdr_packet_decoder, demux(hdr_demuxer)))

hdr_raw = hdr_frame.materialize()
hdr_Y = hdr_raw.planes[0]
print(f"{hdr_raw.pix_fmt = }, {hdr_raw.bit_depth = }, "
      f"{hdr_raw.colorspace = }, {hdr_Y.dtype = }")

# NVDEC surfaces are 16-bit containers holding the samples msb-aligned, so the
# 10 bits sit at the top and the low 6 are zero. Shift them back down to read
# the sample values.
shift = 16 - hdr_raw.bit_depth if device == "cuda" else 0
samples = hdr_Y.to(torch.int32) >> shift
print(f"luma range: [{samples.min()}, {samples.max()}], "
      f"{2 ** hdr_raw.bit_depth} levels available")

# %%
# Streams of unknown length
# -------------------------
#
# :class:`~torchcodec.decoders.VideoDecoder` needs a finite, seekable source:
# it relies on the stream's duration and frame count, and in its default
# ``seek_mode="exact"`` it scans the entire file up-front. The blocks never do
# that - they consume packets as they arrive - so they can decode a source
# that has no duration, no frame count, and no end.
#
# Let's make one: FFmpeg generating frames forever into a named pipe.
import os

fifo_path = temp_dir / "live.ts"
os.mkfifo(fifo_path)


def start_live_stream():
    return subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=640x480:rate=30",  # no duration!
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-g", "30", "-f", "mpegts", "-y", str(fifo_path),
        ],
    )


# %%
# ``VideoDecoder`` can't do anything with that (we ask for the approximate
# seek mode; the exact one would scan the stream forever):
from torchcodec.decoders import VideoDecoder

ffmpeg = start_live_stream()
try:
    VideoDecoder(fifo_path, seek_mode="approximate")
except Exception as e:
    print(f"{type(e).__name__}: {str(e).splitlines()[0]}")
ffmpeg.kill()
ffmpeg.wait()

# %%
# The blocks just stream it, and we stop whenever we want:
ffmpeg = start_live_stream()
demuxer = Demuxer(fifo_path)
packet_decoder = PacketDecoder(demuxer, device=device)
color_converter = ColorConverter(device=device)

frames = []
for frame in color_convert(color_converter, decode(packet_decoder, demux(demuxer))):
    frames.append(frame)
    if len(frames) == 100:
        break  # the stream is still going; we're the ones walking away

print(f"{len(frames)} frames, from pts {frames[0].pts_seconds:.2f}s to "
      f"{frames[-1].pts_seconds:.2f}s, {frames[0].data.shape = }")

ffmpeg.kill()
ffmpeg.wait()
