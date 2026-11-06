# AudioPacketDecoder

*class*torchcodec.decoders.AudioPacketDecoder(*codec_parameters: [AudioCodecParameters](torchcodec.decoders.AudioCodecParameters.html#torchcodec.decoders.AudioCodecParameters)*)[[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#AudioPacketDecoder)

Decodes the compressed [`Packet`](torchcodec.decoders.Packet.html#torchcodec.decoders.Packet)s of one audio stream into
[`RawAudioSamples`](torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples).

This is a low-level API: for straightforward decoding, use
[`AudioDecoder`](torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) instead.

Build it from the [`AudioCodecParameters`](torchcodec.decoders.AudioCodecParameters.html#torchcodec.decoders.AudioCodecParameters) of the stream to decode,
then feed it one packet at a time, and drain it at the end:

```
demuxer = Demuxer("audio.mp3", streams="audio")
decoder = AudioPacketDecoder(demuxer.streams[0].codec_parameters)

for packet in demuxer:
 for raw_samples in decoder.decode(packet):
 ...
for raw_samples in decoder.drain():
 ...
```

The codec parameters can also be pickled, to decode in another process
than the demuxer, or built by hand, to decode packets that don't come from
a [`Demuxer`](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer) (see
[Decoding packets that don't come from a Demuxer](../generated_examples/low_level/demuxer_free.html#sphx-glr-generated-examples-low-level-demuxer-free-py)). Audio is
always decoded on the CPU.

It is stateful. A lossy codec carries state from one frame to the next, so
it expects the packets of its own stream, in the order the demuxer produced
them. After a [`Demuxer.seek()`](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer.seek), `reset()` is necessary but not
sufficient: the first samples that come out are subtly wrong until the codec
re-primes, so decode a margin before your target and throw it away.

Parameters:

**codec_parameters** ([*AudioCodecParameters*](torchcodec.decoders.AudioCodecParameters.html#torchcodec.decoders.AudioCodecParameters)) - The parameters of the stream
to decode, e.g. [`AudioStream.codec_parameters`](torchcodec.decoders.AudioStream.html#torchcodec.decoders.AudioStream.codec_parameters).

Examples using `AudioPacketDecoder`:

![](../_images/sphx_glr_basics_thumb.jpg)

[Build your own decoding pipeline](../generated_examples/low_level/basics.html)

Build your own decoding pipeline
![](../_images/sphx_glr_raw_data_thumb.jpg)

[Raw frames and raw audio samples](../generated_examples/low_level/raw_data.html)

Raw frames and raw audio samples

decode(*packet: [Packet](torchcodec.decoders.Packet.html#torchcodec.decoders.Packet)*) → [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[RawAudioSamples](torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples)][[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#AudioPacketDecoder.decode)

Send one [`Packet`](torchcodec.decoders.Packet.html#torchcodec.decoders.Packet) to the codec and return the
[`RawAudioSamples`](torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples) that are ready.

**This can return zero, one, or more than one**
[`RawAudioSamples`](torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples). What comes back is not the decoding of the
packet you just passed: a codec that is still priming itself will emit
what it owes you on a later call.

Parameters:

**packet** ([*Packet*](torchcodec.decoders.Packet.html#torchcodec.decoders.Packet)) - A packet of this decoder's own stream.

Returns:

The possibly empty list of [`RawAudioSamples`](torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples) that the codec
has ready, in presentation order.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - If this decoder has been drained, or if the demuxer
 seeked without it being `reset()` afterwards.

drain() → [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[RawAudioSamples](torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples)][[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#AudioPacketDecoder.drain)

Tell the codec the stream has ended, and return the
[`RawAudioSamples`](torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples) it was still holding.

Skipping this loses the tail of the stream. A drained decoder refuses
any further packet; `reset()` makes it usable again.

Returns:

The possibly empty list of [`RawAudioSamples`](torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples) that the codec
was still holding, in presentation order.

reset() → [None](https://docs.python.org/3/builtins/constants.html#None)

Drop the codec's buffered state and start over.

Needed after a [`Demuxer.seek()`](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer.seek), and after `drain()`.