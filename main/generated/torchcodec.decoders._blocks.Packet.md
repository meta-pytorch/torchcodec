# Packet

*class*torchcodec.decoders._blocks.Packet(*handle: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *stream_index: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *generation: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*)[[source]](../_modules/torchcodec/decoders/_blocks/_frame.html#Packet)

One compressed packet of one stream, as a [`Demuxer`](torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer) produced it.

You should not build one yourself: a `Demuxer` creates them, and a
[`VideoPacketDecoder`](torchcodec.decoders._blocks.VideoPacketDecoder.html#torchcodec.decoders._blocks.VideoPacketDecoder) or an [`AudioPacketDecoder`](torchcodec.decoders._blocks.AudioPacketDecoder.html#torchcodec.decoders._blocks.AudioPacketDecoder) consumes
them. The contents are opaque: a `Packet` is a handle to an FFmpeg packet.

Examples using `Packet`:

![](../_images/sphx_glr_basics_thumb.png)

[Blocks: build your own decoding pipeline](../generated_examples/blocks/basics.html)

Blocks: build your own decoding pipeline

stream_index*: [int](https://docs.python.org/3/builtins/functions.html#int)*

The index of the stream this packet belongs to, absolute across all
media types. When a demuxer follows more than one stream, this is what
routes each packet to the right decoder.