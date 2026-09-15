# Demuxer

*class*torchcodec.decoders._blocks.Demuxer(*source: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [RawIOBase](https://docs.python.org/3/library/io.html#io.RawIOBase) | BufferedReader*, ***, *streams: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int), ...] = 'video'*)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#Demuxer)

Reads one or more video and audio streams from a container, and produces their compressed [`Packet`](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet)s.

Packets come out interleaved, and `Packet.stream_index` says which
stream each one belongs to:

```
demuxer = Demuxer("video.mp4", streams=("video", "audio"))
decoders = {s.index: s.make_decoder() for s in demuxer.streams}

for packet in demuxer:
 for output in decoders[packet.stream_index].decode(packet):
 ...
```

Parameters:

- **source** (str, `Pathlib.path`, bytes, `torch.Tensor` or file-like object) - 

The source of the media:

- If `str`: a local path or a URL to a media file.
- If `Pathlib.path`: a path to a local media file.
- If `bytes` object or `torch.Tensor`: the raw encoded data.
- If file-like object: we read data from the object on demand. The
object must expose the methods read(self, size: int) -> bytes
and seek(self, offset: int, whence: int) -> int.
- **streams** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*,**optional*) - Which streams to follow, as a
single selector or a tuple of them. A selector is either
`"video"` or `"audio"` for the [best stream](../glossary.html#term-best-stream) of that
type, or an `int` for a stream index, absolute across all media
types. `"all"` follows every audio and video stream in container
order, skipping the rest, and can only be used on its own. Default:
`"video"`.

Variables:

- **streams** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) - The [`VideoStream`](torchcodec.decoders._blocks.VideoStream.html#torchcodec.decoders._blocks.VideoStream) and [`AudioStream`](torchcodec.decoders._blocks.AudioStream.html#torchcodec.decoders._blocks.AudioStream)
objects being followed, in the order the `streams` parameter
named them. Packet decoders are built from these.
- **metadata** ([*DemuxerMetadata*](torchcodec.decoders._blocks.DemuxerMetadata.html#torchcodec.decoders._blocks.DemuxerMetadata)) - What the container header says about the
container itself. What it says about a given stream is on
`demuxer.streams[i].metadata`.

Examples using `Demuxer`:

![](../_images/sphx_glr_blocks_thumb.png)

[Blocks: build your own decoding pipeline](../generated_examples/decoding/blocks.html)

Blocks: build your own decoding pipeline

next_packet() → [Packet](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet) | [None](https://docs.python.org/3/builtins/constants.html#None)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#Demuxer.next_packet)

Read and return the next [`Packet`](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet).

Packets come out interleaved across the streams being followed, in the
order the container stores them, so this is where
`Packet.stream_index` matters: it is what routes each packet to
the decoder of its own stream. Iterating over a `Demuxer` calls this
until it returns `None`.

`None` means the *container* is exhausted, not a stream: it only
comes once no followed stream has a packet left. An individual stream
usually runs dry before that - an audio stream shorter than the video
it accompanies simply stops appearing - and nothing announces that it
did. Its decoder is finished off with `drain()`, not by watching for
`None`. Once exhausted, further calls keep returning `None`; a read
error raises instead.

Returns:

The next packet, or `None` once the container is
exhausted.

Return type:

[Packet](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet) or None

seek(*seconds: [float](https://docs.python.org/3/builtins/functions.html#float)*, ***, *stream: [VideoStream](torchcodec.decoders._blocks.VideoStream.html#torchcodec.decoders._blocks.VideoStream) | [AudioStream](torchcodec.decoders._blocks.AudioStream.html#torchcodec.decoders._blocks.AudioStream) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [None](https://docs.python.org/3/builtins/constants.html#None)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#Demuxer.seek)

Move the demuxer to `seconds`.

This moves *every* stream being followed. For videos, this lands on the
keyframe at or before `seconds`. For audio, a lossy codec's first
frames after a seek are typically slightly wrong until the codec
re-primes. This is especially true when resampling is involved (via an
[`AudioConverter`](torchcodec.decoders._blocks.AudioConverter.html#torchcodec.decoders._blocks.AudioConverter)). Pre-rolling a margin of audio before the
target is up to you.

Important

You must call [`VideoPacketDecoder.reset()`](torchcodec.decoders._blocks.VideoPacketDecoder.html#torchcodec.decoders._blocks.VideoPacketDecoder.reset) or
[`AudioPacketDecoder.reset()`](torchcodec.decoders._blocks.AudioPacketDecoder.html#torchcodec.decoders._blocks.AudioPacketDecoder.reset) on every decoder fed by this
demuxer afterwards, and [`AudioConverter.reset()`](torchcodec.decoders._blocks.AudioConverter.html#torchcodec.decoders._blocks.AudioConverter.reset) on every
converter too: a seek invalidates a codec and resampler states.

Parameters:

- **seconds** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) - The position to seek to.
- **stream** ([*VideoStream*](torchcodec.decoders._blocks.VideoStream.html#torchcodec.decoders._blocks.VideoStream)*or*[*AudioStream*](torchcodec.decoders._blocks.AudioStream.html#torchcodec.decoders._blocks.AudioStream)*,**optional*) - The stream the
target `seconds` is resolved against. FFmpeg resolves a seek in a single
stream's time base and lands on *that* stream's keyframes, the
other streams merely resuming from wherever the container ends
up - so a second video stream may land mid-GOP and decode
garbage until its next keyframe. Defaults to the first of
`streams`, as passed to the constructor.