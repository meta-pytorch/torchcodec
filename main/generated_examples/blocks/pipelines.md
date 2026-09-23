# Multi-threaded decoding pipelines

Warning

**The Blocks APIs are under active construction.** They are private
and unreleased. Signatures and semantics may change without notice. This
tutorial only exists to show what they will eventually make possible.

In this tutorial, we'll assemble the three decoding stages into pipelines of our
own: running demuxing, decoding and color-conversion concurrently on several
threads, choosing where to split them. Each of these steps individually release
the GIL.

Important

The Blocks objects can cross threads, but not processes, so multi-processing
is currently not supported. But it *can* be: if that's something you need,
please open an issue.

Some boilerplate first: a test video, and the device we'll run on.

```
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
```

```
device = 'cuda'

CompletedProcess(args=['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=30:duration=5', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-g', '30', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '/tmp/tmpj__oxlj1/video.mp4'], returncode=0)
```

## One stage, one generator

Each stage can be written as a generator: one over the [`Packet`](../../generated/torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet) objects
a [`Demuxer`](../../generated/torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer) produces, one over the [`RawFrame`](../../generated/torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame) objects a
[`VideoPacketDecoder`](../../generated/torchcodec.decoders._blocks.VideoPacketDecoder.html#torchcodec.decoders._blocks.VideoPacketDecoder) decodes them into, and one over the RGB
[`Frame`](../../generated/torchcodec.Frame.html#torchcodec.Frame) objects a [`ColorConverter`](../../generated/torchcodec.decoders._blocks.ColorConverter.html#torchcodec.decoders._blocks.ColorConverter) makes of those. A
pipeline is then a chain of generators, and inserting `prefetch()` between
two of them puts everything upstream on its own thread: the stages run
concurrently, and since the blocks release the GIL, that's real parallelism.

```
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
 # bounded queue.
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
```

## Which stage to parallelize

With those in hand, a pipeline is one expression, and moving the thread
boundary is moving one `prefetch()` call.

```
def sequential(device):
 # demux -> decode -> color-convert, all on the calling thread.
 demuxer = Demuxer(video_path)
 packet_decoder = demuxer.streams[0].make_decoder(device=device)
 color_converter = ColorConverter(device=device)
 return color_convert(color_converter, decode(packet_decoder, demux(demuxer)))

def convert_on_own_thread(device):
 # [demux + decode] on one thread || [color-convert] on another.
 demuxer = Demuxer(video_path)
 packet_decoder = demuxer.streams[0].make_decoder(device=device)
 color_converter = ColorConverter(device=device)
 raw_frames = prefetch(decode(packet_decoder, demux(demuxer)))
 return color_convert(color_converter, raw_frames)

def demux_on_own_thread(device):
 # [demux] on one thread || [decode + color-convert] on another.
 demuxer = Demuxer(video_path)
 packet_decoder = demuxer.streams[0].make_decoder(device=device)
 color_converter = ColorConverter(device=device)
 packets = prefetch(demux(demuxer))
 return color_convert(color_converter, decode(packet_decoder, packets))

def one_thread_each(device):
 # [demux] || [decode] || [color-convert], a thread per stage.
 demuxer = Demuxer(video_path)
 packet_decoder = demuxer.streams[0].make_decoder(device=device)
 color_converter = ColorConverter(device=device)
 packets = prefetch(demux(demuxer))
 raw_frames = prefetch(decode(packet_decoder, packets))
 return color_convert(color_converter, raw_frames)

PIPELINES = (sequential, convert_on_own_thread, demux_on_own_thread, one_thread_each)
for pipeline in PIPELINES:
 frames = list(pipeline(device))
 print(f"{pipeline.__name__}: {len(frames)} frames on {frames[0].data.device}")
```

```
sequential: 150 frames on cuda:0
convert_on_own_thread: 150 frames on cuda:0
demux_on_own_thread: 150 frames on cuda:0
one_thread_each: 150 frames on cuda:0
```

Which split is best depends on where the work is:

- On the **CPU**, color conversion typically costs about as much as decoding,
so `convert_on_own_thread` is usually the best one (see benchmarks below)
- On **CUDA**, color conversion is comparatively much cheaper and is dwarfed
by the decoding time, so demuxing in parallel with `demux_on_own_thread`
may be the better split.

Let's compare the speedup that `convert_on_own_thread` on the CPU, vs the
sequential pipeline and the [`VideoDecoder.get_all_frames() #`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder.get_all_frames) method as baselines.

```
from time import perf_counter_ns

from torchcodec.decoders import VideoDecoder

def bench(f, num_exp=3, warmup=1):
 for _ in range(warmup):
 f()
 times = []
 for _ in range(num_exp):
 start = perf_counter_ns()
 f()
 times.append(perf_counter_ns() - start)
 return torch.tensor(times).float().median().item() / 1e9

def decode_all_with_videodecoder():
 decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
 return decoder.get_all_frames()

baseline = bench(decode_all_with_videodecoder)
print(f"{'VideoDecoder.get_all_frames()':<29}: {baseline:.2f}s")

for pipeline in (sequential, convert_on_own_thread):
 seconds = bench(lambda p=pipeline: list(p("cpu")))
 print(f"{pipeline.__name__:<29}: {seconds:.2f}s "
 f"({baseline / seconds:.2f}x vs VideoDecoder)")
```

```
VideoDecoder.get_all_frames(): 0.63s
sequential : 0.46s (1.35x vs VideoDecoder)
convert_on_own_thread : 0.40s (1.57x vs VideoDecoder)
```

`sequential` lands on the baseline, as expected: the same work, in the same
order, on one thread. `convert_on_own_thread` is where the speedup is,
because it can overalp the two most expensive steps: decoding and
color-conversion.

**Total running time of the script:** (0 minutes 7.308 seconds)

[`Download Jupyter notebook: pipelines.ipynb`](../../_downloads/1bd9c6e945e662bafe88b1687e2dfa61/pipelines.ipynb)

[`Download Python source code: pipelines.py`](../../_downloads/fceecc295c376d1e6a796f113671556c/pipelines.py)

[`Download zipped: pipelines.zip`](../../_downloads/9d88fed5664210efbeac7b0273765718/pipelines.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)