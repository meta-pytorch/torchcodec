# AudioStream

*class*torchcodec.decoders.AudioStream(*demuxer: [Demuxer](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer)*, *index: [int](https://docs.python.org/3/builtins/functions.html#int)*)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#AudioStream)

An audio stream followed by a [`Demuxer`](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer).

You should not build one yourself: a `Demuxer` creates one for you.

Variables:

**index** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - The stream's index within the container, absolute across
all media types. This is what a [`Packet`](torchcodec.decoders.Packet.html#torchcodec.decoders.Packet)'s `stream_index`
can be compared against.

Examples using `AudioStream`:

![](../_images/sphx_glr_basics_thumb.png)

[Build your own decoding pipeline](../generated_examples/low_level/basics.html)

Build your own decoding pipeline

make_decoder() → [AudioPacketDecoder](torchcodec.decoders.AudioPacketDecoder.html#torchcodec.decoders.AudioPacketDecoder)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#AudioStream.make_decoder)

Build the [`AudioPacketDecoder`](torchcodec.decoders.AudioPacketDecoder.html#torchcodec.decoders.AudioPacketDecoder) for this stream.

Returns:

A decoder for this stream's packets.

Return type:

[AudioPacketDecoder](torchcodec.decoders.AudioPacketDecoder.html#torchcodec.decoders.AudioPacketDecoder)

*property*metadata*: [AudioStreamHeaderMetadata](torchcodec.decoders.AudioStreamHeaderMetadata.html#torchcodec.decoders.AudioStreamHeaderMetadata)*

What the container header says about this audio stream.

From the header only.