# VideoStream

*class*torchcodec.decoders._blocks.VideoStream(*demuxer: [Demuxer](torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer)*, *index: [int](https://docs.python.org/3/builtins/functions.html#int)*)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#VideoStream)

A video stream followed by a [`Demuxer`](torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer).

You should not build one yourself: a `Demuxer` creates one for you.

Variables:

**index** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - The stream's index within the container, absolute across
all media types. This is what a [`Packet`](torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet)'s `stream_index`
can be compared against.

Examples using `VideoStream`:

![](../_images/sphx_glr_basics_thumb.png)

[Blocks: build your own decoding pipeline](../generated_examples/blocks/basics.html)

Blocks: build your own decoding pipeline

make_decoder(*device: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [VideoPacketDecoder](torchcodec.decoders._blocks.VideoPacketDecoder.html#torchcodec.decoders._blocks.VideoPacketDecoder)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#VideoStream.make_decoder)

Build the [`VideoPacketDecoder`](torchcodec.decoders._blocks.VideoPacketDecoder.html#torchcodec.decoders._blocks.VideoPacketDecoder) for this stream.

Parameters:

**device** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device to decode on (cpu or CUDA).
If `None` (default), the current default device is used (see
`torch.set_default_device`).

Returns:

A decoder for this stream's packets.

Return type:

[VideoPacketDecoder](torchcodec.decoders._blocks.VideoPacketDecoder.html#torchcodec.decoders._blocks.VideoPacketDecoder)

*property*metadata*: [VideoStreamHeaderMetadata](torchcodec.decoders._blocks.VideoStreamHeaderMetadata.html#torchcodec.decoders._blocks.VideoStreamHeaderMetadata)*

What the container header says about this video stream.

From the header only. This stream's exact frame count, timestamps and
keyframe positions aren't in there - those come from `scan()`, and
that is the distinction the `_from_header` and `_from_content`
suffixes mark.

scan() → [FrameIndex](torchcodec.decoders._blocks.FrameIndex.html#torchcodec.decoders._blocks.FrameIndex)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#VideoStream.scan)

Demux this stream from end to end, without decoding it (a
[scan](../glossary.html#term-scan)), and return its [`FrameIndex`](torchcodec.decoders._blocks.FrameIndex.html#torchcodec.decoders._blocks.FrameIndex).

This is the only way to know a stream's exact frame count, timestamps
and keyframe positions: the container header can be wrong about all
three.

You can call this on multiple video streams from the same demuxer. The
first call scans the entire video once, and subsequent calls on other
streams are free.

Important

If called, then this must be called before any packets are read from
the demuxer.

Returns:

the frame index for this stream, describing its keyframes and frame positions.

Return type:

[FrameIndex](torchcodec.decoders._blocks.FrameIndex.html#torchcodec.decoders._blocks.FrameIndex)