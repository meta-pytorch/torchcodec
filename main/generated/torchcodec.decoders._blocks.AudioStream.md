# AudioStream

*class*torchcodec.decoders._blocks.AudioStream(*demuxer: [Demuxer](torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer)*, *index: [int](https://docs.python.org/3/builtins/functions.html#int)*)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#AudioStream)

An audio stream followed by a [`Demuxer`](torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer).

You should not build one yourself: a `Demuxer` creates one for you.

Variables:

**index** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - The stream's index within the container, absolute across
all media types. This is what a [`Packet`](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet)'s `stream_index`
can be compared against.

Examples using `AudioStream`:

![](../_images/sphx_glr_blocks_thumb.png)

[Blocks: build your own decoding pipeline](../generated_examples/decoding/blocks.html)

Blocks: build your own decoding pipeline

make_decoder() → [AudioPacketDecoder](torchcodec.decoders._blocks.AudioPacketDecoder.html#torchcodec.decoders._blocks.AudioPacketDecoder)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#AudioStream.make_decoder)

Build the [`AudioPacketDecoder`](torchcodec.decoders._blocks.AudioPacketDecoder.html#torchcodec.decoders._blocks.AudioPacketDecoder) for this stream.

Returns:

A decoder for this stream's packets.

Return type:

[AudioPacketDecoder](torchcodec.decoders._blocks.AudioPacketDecoder.html#torchcodec.decoders._blocks.AudioPacketDecoder)

*property*metadata*: [AudioStreamHeaderMetadata](torchcodec.decoders._blocks.AudioStreamHeaderMetadata.html#torchcodec.decoders._blocks.AudioStreamHeaderMetadata)*

What the container header says about this audio stream.

From the header only.