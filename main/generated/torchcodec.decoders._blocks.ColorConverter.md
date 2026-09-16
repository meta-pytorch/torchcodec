# ColorConverter

*class*torchcodec.decoders._blocks.ColorConverter(*device: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *output_dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['auto'] = torch.uint8*)[[source]](../_modules/torchcodec/decoders/_blocks/_color_converter.html#ColorConverter)

Turn a [`RawFrame`](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame) (typically YUV) into an RGB [`Frame`](torchcodec.Frame.html#torchcodec.Frame).

```
converter = ColorConverter()

for packet in demuxer:
 for raw_frame in packet_decoder.decode(packet):
 frame = converter.convert(raw_frame)
 frame.data # uint8 [3, height, width], RGB
```

Unlike the other blocks this one isn't tied to a specific video stream.
Everything it needs (dimensions, pixel format, colorspace, rotation) comes
from the [`RawFrame`](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame) itself, so the same converter instance can
process frames from any video stream, provided that they share the same
device.

Parameters:

- **device** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device to convert on. If
`None` (default), the current default device is used (see
`torch.set_default_device`). It has to be the device the frames
are already on, i.e. it must match what was passed to the
[`VideoPacketDecoder`](torchcodec.decoders._blocks.VideoPacketDecoder.html#torchcodec.decoders._blocks.VideoPacketDecoder) that produced
the [`RawFrame`](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame).
- **output_dtype** (torch.dtype or `"auto"`, optional) - `torch.uint8`
(default) for values in `[0, 255]`, `torch.float32` for
`[0, 1]`, or `"auto"` for uint8 from 8-bit sources and float32
from deeper ones. Since this block isn't tied to a stream,
`"auto"` is resolved per frame rather than once per video, so
feeding it a mix of SDR and HDR frames gives you a mix of dtypes.

Examples using `ColorConverter`:

![](../_images/sphx_glr_basics_thumb.png)

[Blocks: build your own decoding pipeline](../generated_examples/blocks/basics.html)

Blocks: build your own decoding pipeline
![](../_images/sphx_glr_pipelines_thumb.png)

[Composing pipelines: threads, devices and endless streams](../generated_examples/blocks/pipelines.html)

Composing pipelines: threads, devices and endless streams
![](../_images/sphx_glr_raw_data_thumb.png)

[Raw frames and raw audio samples](../generated_examples/blocks/raw_data.html)

Raw frames and raw audio samples

convert(*raw_frame: [RawFrame](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame)*) → [Frame](torchcodec.Frame.html#torchcodec.Frame)[[source]](../_modules/torchcodec/decoders/_blocks/_color_converter.html#ColorConverter.convert)

Convert one [`RawFrame`](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame) to an RGB [`Frame`](torchcodec.Frame.html#torchcodec.Frame).

[`RawFrame.rotation_degrees`](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame.rotation_degrees) is applied, so the output is upright
and matches what a [`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) gives you.

Parameters:

**raw_frame** ([*RawFrame*](torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame)) - The frame to convert. It has to be on this
converter's device.

Returns:

The RGB `[3, height, width]` frame in the converter's
`output_dtype`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - If the frame is not on this converter's device.