# VideoPacketDecoder

*class*torchcodec.decoders._blocks.VideoPacketDecoder(**args*, ***kwargs*)[[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#VideoPacketDecoder)

Decodes the compressed [`Packet`](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet)s of one video stream into
[`RawFrame`](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame)s.

You should not build one yourself: [`VideoStream.make_decoder()`](torchcodec.decoders._blocks.VideoStream.html#torchcodec.decoders._blocks.VideoStream.make_decoder) is what
creates it. Frames come out on the device given there.

It is stateful. It holds the codec's reference-frame buffer, so it expects
the packets of its own stream, in the order the demuxer produced them.

Examples using `VideoPacketDecoder`:

![](../_images/sphx_glr_blocks_thumb.png)

[Blocks: build your own decoding pipeline](../generated_examples/decoding/blocks.html)

Blocks: build your own decoding pipeline

decode(*packet: [Packet](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet)*) → [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[RawFrame](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame)][[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#VideoPacketDecoder.decode)

Send one [`Packet`](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet) to the codec and return the
[`RawFrame`](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame)s that are ready.

**This can return zero, one, or more than one** [`RawFrame`](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame). What
comes back is not the decoding of the packet you just passed: a codec
that is buffering B-frames, or still priming itself, will emit what it owes
you on a later call.

Parameters:

**packet** ([*Packet*](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet)) - A packet of this decoder's own stream.

Returns:

The possibly empty list of [`RawFrame`](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame)s that the codec has
ready, in presentation order.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - If this decoder has been drained, or if the demuxer
 seeked without it being `reset()` afterwards.

drain() → [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[RawFrame](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame)][[source]](../_modules/torchcodec/decoders/_blocks/_packet_decoder.html#VideoPacketDecoder.drain)

Tell the codec the stream has ended, and return the
[`RawFrame`](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame)s it was still holding.

Skipping this loses the tail of the stream. A drained decoder refuses
any further packet; `reset()` makes it usable again.

Returns:

The possibly empty list of [`RawFrame`](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame)s that the codec was
still holding, in presentation order.

reset() → [None](https://docs.python.org/3/builtins/constants.html#None)

Drop the codec's buffered state and start over.

Needed after a [`Demuxer.seek()`](torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer.seek), and after `drain()`.