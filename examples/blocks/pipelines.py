# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
=========================================================
Composing pipelines: threads, devices and endless streams
=========================================================

How to overlap the decoding stages across threads, choose where to cut the
pipeline on CPU and on CUDA, and decode a source that never ends.

.. warning::

   **The Blocks APIs are under active construction.** They are private
   and unreleased. Signatures and semantics may change without notice. This
   tutorial only exists to show what they will eventually make possible.

:ref:`sphx_glr_generated_examples_blocks_basics.py` ran the three stages -
:class:`Demuxer`, :class:`VideoPacketDecoder`, :class:`ColorConverter` - back
to back on the calling thread. That is the one pipeline shape
:class:`~torchcodec.decoders.VideoDecoder` could have given you as well. This
tutorial is about the ones it couldn't: running the stages concurrently,
putting the thread boundary where your hardware wants it, wrapping the whole
thing in an object of your own, and decoding a source that has no end.

All of that is possible because the blocks are *passive*. They never create a
thread, never own a thread pool, and never decide when work happens: a block
only does something when you call into it. And because each one releases the
GIL while it is in C++, running two of them on two Python threads is real
parallelism, not interleaving.

.. currentmodule:: torchcodec.decoders._blocks
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
# One stage, one generator
# ------------------------
#
# Each stage is a generator: one over the :class:`Packet` objects a
# :class:`Demuxer` produces, one over the :class:`RawFrame` objects a
# :class:`VideoPacketDecoder` decodes them into, and one over the RGB
# :class:`~torchcodec.Frame` objects a :class:`ColorConverter` makes of those. A
# pipeline is then a chain of generators, and inserting ``prefetch()`` between
# two of them puts everything upstream on its own thread: the stages run
# concurrently, and since the blocks release the GIL, that's real parallelism.
import queue
import threading

from torchcodec.decoders._blocks import ColorConverter, Demuxer


def demux(demuxer):
    yield from demuxer


def decode(packet_decoder, packets):
    for packet in packets:
        yield from packet_decoder.decode(packet)
    yield from packet_decoder.drain()


def color_convert(color_converter, raw_frames):
    for raw_frame in raw_frames:
        yield color_converter.convert(raw_frame)


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


# %%
# Where to cut
# ------------
#
# With those in hand, a pipeline is one expression, and moving the thread
# boundary is moving one ``prefetch()`` call.
def sequential():
    # demux -> decode -> color-convert, all on the calling thread.
    demuxer = Demuxer(video_path)
    packet_decoder = demuxer.streams[0].make_decoder(device=device)
    color_converter = ColorConverter(device=device)
    return color_convert(color_converter, decode(packet_decoder, demux(demuxer)))


def convert_on_own_thread():
    # [demux + decode] on one thread || [color-convert] on another.
    demuxer = Demuxer(video_path)
    packet_decoder = demuxer.streams[0].make_decoder(device=device)
    color_converter = ColorConverter(device=device)
    raw_frames = prefetch(decode(packet_decoder, demux(demuxer)))
    return color_convert(color_converter, raw_frames)


def demux_on_own_thread():
    # [demux] on one thread || [decode + color-convert] on another.
    demuxer = Demuxer(video_path)
    packet_decoder = demuxer.streams[0].make_decoder(device=device)
    color_converter = ColorConverter(device=device)
    packets = prefetch(demux(demuxer))
    return color_convert(color_converter, decode(packet_decoder, packets))


for pipeline in (sequential, convert_on_own_thread, demux_on_own_thread):
    frames = list(pipeline())
    print(f"{pipeline.__name__}: {len(frames)} frames on {frames[0].data.device}")

# %%
# Which of the two splits is the good one depends on where the work is, and
# that is a property of the device rather than of the file:
#
# * On the **CPU**, all three stages compete for the same cores, and color
#   conversion is the expensive one. Giving it a thread of its own -
#   ``convert_on_own_thread`` - is the split that pays.
# * On **CUDA**, demuxing is CPU and I/O work, while decoding (NVDEC) and color
#   conversion both happen on the GPU. Keeping the two GPU stages together and
#   feeding them from a demuxing thread - ``demux_on_own_thread`` - is the
#   natural shape: the CPU stays ahead of the GPU instead of taking turns with
#   it.
#
# Nothing stops you from doing something else entirely: one pipeline per file,
# decoding on the CPU while color-converting on the GPU, several decoders
# feeding one converter, or frames going straight into your own pre-fetching
# data loader.

# %%
# .. note::
#
#    One CUDA rule comes with this freedom. A :class:`RawFrame` is a view into a
#    buffer the decoder will reuse, so a consumer that reads its samples on a
#    CUDA stream *other* than the one the decoder ran on must call
#    :meth:`RawFrame.record_stream` before the frame goes out of scope. A
#    :class:`ColorConverter` does it for you, so the pipelines above are safe;
#    see
#    :ref:`sphx_glr_generated_examples_blocks_raw_data.py` if you consume the
#    planes yourself.


# %%
# Making it an object
# -------------------
#
# Chained generators are the shortest way to write a pipeline down, not
# necessarily the way you want to ship one. The blocks are meant to be the
# internals of your own class: it owns the three objects, keeps them alive
# together, and exposes whatever interface the rest of your code wants -
# ``__iter__`` here, but a ``torch.utils.data.IterableDataset``, an actor, or a
# ``next_batch()`` method are all the same handful of lines.
class VideoPipeline:
    def __init__(self, path, *, device=None, prefetch_packets=True):
        self._demuxer = Demuxer(path)
        (stream,) = self._demuxer.streams
        self._packet_decoder = stream.make_decoder(device=device)
        self._color_converter = ColorConverter(device=device)
        self._prefetch_packets = prefetch_packets

    def __iter__(self):
        packets = demux(self._demuxer)
        if self._prefetch_packets:
            packets = prefetch(packets)
        raw_frames = decode(self._packet_decoder, packets)
        yield from color_convert(self._color_converter, raw_frames)


frames = list(VideoPipeline(video_path, device=device))
print(f"{len(frames)} frames, up to {frames[-1].pts_seconds:.2f}s")

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
# :class:`~torchcodec.decoders.VideoDecoder` can't do anything with that (we ask
# for the approximate seek mode; the exact one would scan the stream forever):
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
packet_decoder = demuxer.streams[0].make_decoder(device=device)
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

# %%
# Nothing about the pipeline changed to make this work - it is the same three
# generators. Walking away early is just not pulling from them again, and the
# ``prefetch()`` boundaries above compose with it unchanged: the worker thread
# is blocked in ``q.put()`` behind the bounded queue, and it is a daemon, so it
# goes away with the process.
