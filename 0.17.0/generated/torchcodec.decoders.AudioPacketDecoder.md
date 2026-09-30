# AudioPacketDecoder

*class*torchcodec.decoders.AudioPacketDecoder(**args*, ***kwargs*)[[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#AudioPacketDecoder)

Decodes the compressed [`Packet`](torchcodec.decoders.Packet.html#torchcodec.decoders.Packet)s of one audio stream into
[`RawAudioSamples`](torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples).

This is a low-level API: for straightforward decoding, use
[`AudioDecoder`](torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) instead.

You should not build one yourself: [`AudioStream.make_decoder()`](torchcodec.decoders.AudioStream.html#torchcodec.decoders.AudioStream.make_decoder) is what
creates it. Audio is always decoded on the CPU.

Feed it one packet at a time, and drain it at the end:

```
demuxer = Demuxer("audio.mp3", streams="audio")
decoder = demuxer.streams[0].make_decoder()

for packet in demuxer:
 for raw_samples in decoder.decode(packet):
 ...
for raw_samples in decoder.drain():
 ...
```

It is stateful. A lossy codec carries state from one frame to the next, so
it expects the packets of its own stream, in the order the demuxer produced
them. After a [`Demuxer.seek()`](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer.seek), `reset()` is necessary but not
sufficient: the first samples that come out are subtly wrong until the codec
re-primes, so decode a margin before your target and throw it away.

Examples using `AudioPacketDecoder`:

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