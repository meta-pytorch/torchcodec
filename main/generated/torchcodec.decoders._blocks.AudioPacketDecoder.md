# AudioPacketDecoder

*class*torchcodec.decoders._blocks.AudioPacketDecoder(**args*, ***kwargs*)[[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#AudioPacketDecoder)

TODO_API_BREAKDOWN DOC

Examples using `AudioPacketDecoder`:

![](../_images/sphx_glr_blocks_thumb.png)

[Blocks: build your own decoding pipeline](../generated_examples/decoding/blocks.html)

Blocks: build your own decoding pipeline

decode(*packet: [Packet](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet)*) → [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[RawAudioSamples](torchcodec.decoders._blocks.RawAudioSamples.html#torchcodec.decoders._blocks.RawAudioSamples)][[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#AudioPacketDecoder.decode)

Send one [`Packet`](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet) to the codec and return whatever is ready.

The result is often empty, and it is not "the decoding of that packet":
a codec that is buffering B-frames, or still priming itself, emits what
it owes you on a later call.

Parameters:

**packet** ([*Packet*](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet)) - A packet of this decoder's own stream.

Returns:

What the codec had ready, in presentation order. Possibly nothing.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - If this decoder has been drained, or if the demuxer
 seeked without it being `reset()` afterwards.

drain() → [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[RawAudioSamples](torchcodec.decoders._blocks.RawAudioSamples.html#torchcodec.decoders._blocks.RawAudioSamples)][[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#AudioPacketDecoder.drain)

Tell the codec the stream has ended, and return what it was still
holding.

Skipping this loses the tail of the stream. A drained decoder refuses
any further packet; `reset()` makes it usable again.

Returns:

The last of what the codec had buffered, in presentation order.

reset() → [None](https://docs.python.org/3/builtins/constants.html#None)

Drop the codec's buffered state and start over.

Needed after a [`Demuxer.seek()`](torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer.seek), and after `drain()`.