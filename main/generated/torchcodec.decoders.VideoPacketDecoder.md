# VideoPacketDecoder

*class*torchcodec.decoders.VideoPacketDecoder(*codec_parameters: [VideoCodecParameters](torchcodec.decoders.VideoCodecParameters.html#torchcodec.decoders.VideoCodecParameters)*, *device: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *num_ffmpeg_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 1*)[[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#VideoPacketDecoder)

Decodes the compressed [`Packet`](torchcodec.decoders.Packet.html#torchcodec.decoders.Packet)s of one video stream into
[`RawFrame`](torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame)s.

This is a low-level API: for straightforward decoding, use
[`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) instead.

Build it from the [`VideoCodecParameters`](torchcodec.decoders.VideoCodecParameters.html#torchcodec.decoders.VideoCodecParameters) of the stream to decode,
then feed it one packet at a time, and drain it at the end:

```
demuxer = Demuxer("video.mp4")
decoder = VideoPacketDecoder(demuxer.streams[0].codec_parameters)

for packet in demuxer:
 for raw_frame in decoder.decode(packet):
 ...
for raw_frame in decoder.drain():
 ...
```

The codec parameters can also be pickled, to decode in another process
than the demuxer, or built by hand, to decode packets that don't come from
a [`Demuxer`](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer) (see
[Decoding packets that don't come from a Demuxer](../generated_examples/low_level/demuxer_free.html#sphx-glr-generated-examples-low-level-demuxer-free-py)). Frames come
out on the device the decoder was built for.

It is stateful. It holds the codec's reference-frame buffer, so it expects
the packets of its own stream, in the order the demuxer produced them.

Parameters:

- **codec_parameters** ([*VideoCodecParameters*](torchcodec.decoders.VideoCodecParameters.html#torchcodec.decoders.VideoCodecParameters)) - The parameters of the stream
to decode, e.g. [`VideoStream.codec_parameters`](torchcodec.decoders.VideoStream.html#torchcodec.decoders.VideoStream.codec_parameters).
- **device** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device to decode on (cpu or
CUDA). If `None` (default), the current default device is used
(see `torch.set_default_device`). With codec parameters you built
yourself, CUDA needs some fields to be set: see
[`VideoCodecParameters`](torchcodec.decoders.VideoCodecParameters.html#torchcodec.decoders.VideoCodecParameters).
- **num_ffmpeg_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - The number of threads to use for
CPU decoding. This has no effect when decoding on GPU. Use 1 for
single-threaded decoding, which may be best if you are decoding
multiple streams in parallel. Use a higher number for
multi-threaded decoding, which is best for a single stream.
Passing 0 lets FFmpeg decide on the number of threads.
Default: 1.

Examples using `VideoPacketDecoder`:

![](../_images/sphx_glr_basics_thumb.jpg)

[Build your own decoding pipeline](../generated_examples/low_level/basics.html)

Build your own decoding pipeline
![](../_images/sphx_glr_cuda_streams_thumb.jpg)

[Low-level APIs and CUDA streams synchronization](../generated_examples/low_level/cuda_streams.html)

Low-level APIs and CUDA streams synchronization
![](../_images/sphx_glr_demuxer_free_thumb.jpg)

[Decoding packets that don't come from a Demuxer](../generated_examples/low_level/demuxer_free.html)

Decoding packets that don't come from a Demuxer
![](../_images/sphx_glr_pipelines_thumb.jpg)

[Multi-threaded decoding pipelines](../generated_examples/low_level/pipelines.html)

Multi-threaded decoding pipelines
![](../_images/sphx_glr_raw_data_thumb.jpg)

[Raw frames and raw audio samples](../generated_examples/low_level/raw_data.html)

Raw frames and raw audio samples

decode(*packet: [Packet](torchcodec.decoders.Packet.html#torchcodec.decoders.Packet)*) → [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[RawFrame](torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame)][[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#VideoPacketDecoder.decode)

Send one [`Packet`](torchcodec.decoders.Packet.html#torchcodec.decoders.Packet) to the codec and return the
[`RawFrame`](torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame)s that are ready.

**This can return zero, one, or more than one** [`RawFrame`](torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame). What
comes back is not the decoding of the packet you just passed: a codec
that is buffering B-frames, or still priming itself, will emit what it owes
you on a later call.

Parameters:

**packet** ([*Packet*](torchcodec.decoders.Packet.html#torchcodec.decoders.Packet)) - A packet of this decoder's own stream.

Returns:

The possibly empty list of [`RawFrame`](torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame)s that the codec has
ready, in presentation order.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - If this decoder has been drained, or if the demuxer
 seeked without it being `reset()` afterwards.

drain() → [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[RawFrame](torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame)][[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#VideoPacketDecoder.drain)

Tell the codec the stream has ended, and return the
[`RawFrame`](torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame)s it was still holding.

Skipping this loses the tail of the stream. A drained decoder refuses
any further packet; `reset()` makes it usable again.

Returns:

The possibly empty list of [`RawFrame`](torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame)s that the codec was
still holding, in presentation order.

reset() → [None](https://docs.python.org/3/builtins/constants.html#None)

Drop the codec's buffered state and start over.

Needed after a [`Demuxer.seek()`](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer.seek), and after `drain()`.