# Blocks and CUDA streams

Warning

**The Blocks APIs are under active construction.** They are private
and unreleased. Signatures and semantics may change without notice. This
tutorial only exists to show what they will eventually make possible.

On CUDA, if you decode on one stream and consume the [`RawFrame`](../../generated/torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame)s on
another - either with a [`ColorConverter`](../../generated/torchcodec.decoders._blocks.ColorConverter.html#torchcodec.decoders._blocks.ColorConverter), or with your own consumer as in
[Raw frames and raw audio samples](raw_data.html#sphx-glr-generated-examples-blocks-raw-data-py) - you must take care of
two things:

1. Waiting for the decoder's asynchronous copy before reading the samples.
2. Keeping the CUDA caching allocator from reusing the frame's memory while
your reads are still queued.

Getting this wrong will lead to occasional corrupted frames. We cover both
points below.

Note that if you don't create a separate stream for consuming the RawFrame,
none of this applies: everything runs on the same stream and stream ordering
handles it.

Some boilerplate first: a test video, the two streams we'll use, and a helper
to build a fresh set of blocks for each example below.

```
import subprocess
import tempfile
from pathlib import Path

import torch

from torchcodec.decoders._blocks import ColorConverter, Demuxer

video_path = Path(tempfile.mkdtemp()) / "video.mp4"
subprocess.run(
 [
 "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
 "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=2",
 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30",
 str(video_path),
 ],
 check=True,
)

decode_stream = torch.cuda.Stream()
convert_stream = torch.cuda.Stream()

def make_blocks():
 demuxer = Demuxer(video_path)
 packet_decoder = demuxer.streams[0].make_decoder(device="cuda")
 return demuxer, packet_decoder, ColorConverter(device="cuda")
```

## Waiting for the samples to be ready

A [`RawFrame`](../../generated/torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame)'s samples are produced on the stream that was current when
you called [`VideoPacketDecoder.decode()`](../../generated/torchcodec.decoders._blocks.VideoPacketDecoder.html#torchcodec.decoders._blocks.VideoPacketDecoder.decode).
[`VideoPacketDecoder.decode()`](../../generated/torchcodec.decoders._blocks.VideoPacketDecoder.html#torchcodec.decoders._blocks.VideoPacketDecoder.decode) typically involves a CUDA copy of the NVDEC
decoder internals into a PyTorch-allocated CUDA tensor. This copy is
asynchronous, so a consumer on another stream has to wait for it to finish.
For that, you can record an event on the decoding stream and wait on it:

```
with torch.cuda.stream(decode_stream):
 raw_frames = packet_decoder.decode(packet)
 decoded = torch.cuda.Event()
 decoded.record()

decoded.wait(convert_stream)
with torch.cuda.stream(convert_stream):
 frame = color_converter.convert(raw_frames[0])
```

Alternatively, you can use `convert_stream.wait_stream(decode_stream)`, but
it has different semantics: it waits for everything currently queued on the
decoding stream, so a consumer that has fallen behind waits for the decoder's
whole backlog instead of for its own frame.

## Keeping the allocator from reusing the memory

Here what can go wrong:

```
with torch.cuda.stream(decode_stream):
 raw_frames = packet_decoder.decode(packet) # allocated on decode_stream

with torch.cuda.stream(convert_stream):
 frame = color_converter.convert(raw_frames[0]) # reads are only *queued*

del raw_frames # the allocation goes back to decode_stream's pool

with torch.cuda.stream(decode_stream):
 packet_decoder.decode(next_packet) # BAD: the next frame can be given
 # that same memory by the PyTorch CUDA
 # allocator, and overwrite
 # the samples convert() is reading
```

To learn more, see the *Streams and freeing memory* section of [this blog post](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html) .

You have two ways to handle this each with its own trade-offs.

### Option 1: sync back before dropping the frame

Make the decoding stream wait for your reads, then let the frame go:

```
decode_stream.wait_stream(convert_stream)
del raw_frames
```

The issue here is that the decoder's next frame cannot start until your reads
have finished, which removes the overlap you created a second stream for. To
get this overlap back, you can work in batches: hold N frames, sync once, drop
them together. The decoder then stalls once every N frames, and N is an
explicit memory knob:

```
MAX_INFLIGHT_FRAMES = 8

demuxer, packet_decoder, color_converter = make_blocks()

frames, in_flight = [], []
for packet in list(demuxer) + [None]:
 with torch.cuda.stream(decode_stream):
 raw_frames = (
 packet_decoder.drain() if packet is None else packet_decoder.decode(packet)
 )
 decoded = torch.cuda.Event()
 decoded.record()

 decoded.wait(convert_stream)
 with torch.cuda.stream(convert_stream):
 for raw_frame in raw_frames:
 frames.append(color_converter.convert(raw_frame))

 # Hold on to the raw frames, so that their memory can't be reused yet.
 in_flight += raw_frames
 if len(in_flight) >= MAX_INFLIGHT_FRAMES:
 # Make the decoder wait for the conversions we queued, then let this
 # batch of frames and their memory go.
 decode_stream.wait_stream(convert_stream)
 in_flight.clear()

decode_stream.wait_stream(convert_stream)
in_flight.clear()
torch.cuda.synchronize()

print(f"Option 1: {len(frames)} frames, {frames[0].data.shape = }")
```

```
Option 1: 60 frames, frames[0].data.shape = torch.Size([3, 720, 1280])
```

### Option 2: let the allocator handle it

[`torch.Tensor.record_stream()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream) tells the allocator that another stream is
using the block, so it withholds it from reuse until your reads complete. Call
it on [`RawFrame.storage_cuda`](../../generated/torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame.storage_cuda), right after queueing your reads.

This never stalls the decoder, so you keep full overlap, and you pay in memory
instead. How much is decided by device timing rather than by you: the
allocator polls the recorded stream on later allocations, so peak memory
varies between runs and can't be attributed to a line of your code. Close to
your memory ceiling, the allocator's fallback is to block and release cached
blocks, which is a large stall at an unpredictable point.

Note

`record_stream` only works on [`RawFrame.storage_cuda`](../../generated/torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame.storage_cuda). Calling it
on `RawFrame.planes` is silently a no-op

```
demuxer, packet_decoder, color_converter = make_blocks()

frames = []
for packet in list(demuxer) + [None]:
 with torch.cuda.stream(decode_stream):
 raw_frames = (
 packet_decoder.drain() if packet is None else packet_decoder.decode(packet)
 )
 decoded = torch.cuda.Event()
 decoded.record()

 decoded.wait(convert_stream)
 with torch.cuda.stream(convert_stream):
 for raw_frame in raw_frames:
 frames.append(color_converter.convert(raw_frame))
 raw_frame.storage_cuda.record_stream(convert_stream)

torch.cuda.synchronize()

print(f"Option 2: {len(frames)} frames, {frames[0].data.shape = }")
```

```
Option 2: 60 frames, frames[0].data.shape = torch.Size([3, 720, 1280])
```

**Total running time of the script:** (0 minutes 0.461 seconds)

[`Download Jupyter notebook: cuda_streams.ipynb`](../../_downloads/a69034aed2fe1f84ed02ff93bcb31c72/cuda_streams.ipynb)

[`Download Python source code: cuda_streams.py`](../../_downloads/6b0b0dc6bbb33ea64d140d8ad5e3113c/cuda_streams.py)

[`Download zipped: cuda_streams.zip`](../../_downloads/c59cdf6bd944ae42ea18044238c3e753/cuda_streams.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)