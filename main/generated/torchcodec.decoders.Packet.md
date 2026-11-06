# Packet

*class*torchcodec.decoders.Packet(*handle: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *stream_index: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *generation: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*)[[source]](../_modules/torchcodec/decoders/_blocks/_frame.html#Packet)

One compressed packet of one stream.

A [`Demuxer`](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer) creates them, and a [`VideoPacketDecoder`](torchcodec.decoders.VideoPacketDecoder.html#torchcodec.decoders.VideoPacketDecoder) or an
[`AudioPacketDecoder`](torchcodec.decoders.AudioPacketDecoder.html#torchcodec.decoders.AudioPacketDecoder) consumes them. You can also build one from
compressed data that doesn't come from a `Demuxer`, with
`from_bytes()` or `from_tensor()`.

Packets can be pickled, e.g. to be decoded in another process. The pickles
are only guaranteed to be readable by the same version of TorchCodec.

Examples using `Packet`:

![](../_images/sphx_glr_basics_thumb.jpg)

[Build your own decoding pipeline](../generated_examples/low_level/basics.html)

Build your own decoding pipeline
![](../_images/sphx_glr_demuxer_free_thumb.jpg)

[Decoding packets that don't come from a Demuxer](../generated_examples/low_level/demuxer_free.html)

Decoding packets that don't come from a Demuxer

*property*data*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

The compressed data, as a 1D uint8 CPU tensor.

It is a view on the packet's data: writing to it modifies the packet,
so `.clone()` it first.

*property*dts*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The decoding timestamp, in units of the stream's time base. `None`
if unknown.

*property*duration*: [int](https://docs.python.org/3/builtins/functions.html#int)*

How long the packet lasts, in units of the stream's time base. 0 if
unknown.

*property*flags*: [int](https://docs.python.org/3/builtins/functions.html#int)*

FFmpeg's `AV_PKT_FLAG_*` bits, e.g. `flags & 1` is set for
keyframes.

*classmethod*from_bytes(*data: [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [bytearray](https://docs.python.org/3/builtins/stdtypes.html#bytearray) | [memoryview](https://docs.python.org/3/builtins/stdtypes.html#memoryview)*, ***, *pts: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *dts: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *duration: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *flags: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *side_data: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *stream_index: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*) → Packet[[source]](../_modules/torchcodec/decoders/_blocks/_frame.html#Packet.from_bytes)

Build a packet from compressed data, e.g. read from a database or
received from the network.

Only `data` is required: most codecs need nothing else to decode it.

Parameters:

- **data** ([*bytes*](https://docs.python.org/3/builtins/stdtypes.html#bytes)*,*[*bytearray*](https://docs.python.org/3/builtins/stdtypes.html#bytearray)*or*[*memoryview*](https://docs.python.org/3/builtins/stdtypes.html#memoryview)) - The compressed data of one
packet. It is copied.
- **pts** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - The [pts](../glossary.html#term-pts), in units of the stream's time
base ([`StreamMetadata.time_base`](torchcodec.decoders.StreamMetadata.html#torchcodec.decoders.StreamMetadata.time_base)). The decoded frames'
[`RawFrame.pts`](torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame.pts) come from it.
`None` (default) if unknown.
- **dts** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - The decoding timestamp, in units of the
stream's time base. `None` (default) if unknown.
- **duration** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - How long the packet lasts, in units of
the stream's time base. Default: 0, for unknown.
- **flags** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - FFmpeg's `AV_PKT_FLAG_*` bits, e.g. 1
(`AV_PKT_FLAG_KEY`) for a keyframe. `AV_PKT_FLAG_TRUSTED`
(8) isn't allowed. Default: 0.
- **side_data** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - Extra data about the packet, keyed by
FFmpeg's names for its types, e.g. `"Skip Samples"`.
- **stream_index** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - See `stream_index`. Default: 0.

Returns:

The packet.

Return type:

Packet

*classmethod*from_tensor(*data: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *pts: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *dts: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *duration: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *flags: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *side_data: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *stream_index: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*) → Packet[[source]](../_modules/torchcodec/decoders/_blocks/_frame.html#Packet.from_tensor)

Build a packet from compressed data in a tensor.

Same as `from_bytes()`, except for `data`.

Parameters:

- **data** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - The compressed data of one packet, as a
contiguous 1D uint8 CPU tensor. It is copied.
- **pts** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - See `from_bytes()`.
- **dts** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - See `from_bytes()`.
- **duration** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - See `from_bytes()`.
- **flags** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - See `from_bytes()`.
- **side_data** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - See `from_bytes()`.
- **stream_index** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - See `from_bytes()`.

Returns:

The packet.

Return type:

Packet

*property*pts*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The [pts](../glossary.html#term-pts), in units of the stream's time base
([`StreamMetadata.time_base`](torchcodec.decoders.StreamMetadata.html#torchcodec.decoders.StreamMetadata.time_base)). `None` if unknown.

*property*side_data*: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes)]*

Extra data about the packet, keyed by FFmpeg's names for its types,
e.g. `"Skip Samples"`.

stream_index*: [int](https://docs.python.org/3/builtins/functions.html#int)*

The index of the stream this packet belongs to, absolute across all
media types. When a demuxer follows more than one stream, this is what
routes each packet to the right decoder.