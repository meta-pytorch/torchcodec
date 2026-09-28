# Packet

*class*torchcodec.decoders.Packet(*handle: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *stream_index: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *generation: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*)[[source]](../_modules/torchcodec/decoders/_blocks/_frame.html#Packet)

One compressed packet of one stream, as a [`Demuxer`](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer) produced it.

You should not build one yourself: a `Demuxer` creates them, and a
[`VideoPacketDecoder`](torchcodec.decoders.VideoPacketDecoder.html#torchcodec.decoders.VideoPacketDecoder) or an [`AudioPacketDecoder`](torchcodec.decoders.AudioPacketDecoder.html#torchcodec.decoders.AudioPacketDecoder) consumes
them. The contents are opaque: a `Packet` is a handle to an FFmpeg packet.

Examples using `Packet`:

![](../_images/sphx_glr_basics_thumb.png)

[Build your own decoding pipeline](../generated_examples/low_level/basics.html)

Build your own decoding pipeline

stream_index*: [int](https://docs.python.org/3/builtins/functions.html#int)*

The index of the stream this packet belongs to, absolute across all
media types. When a demuxer follows more than one stream, this is what
routes each packet to the right decoder.