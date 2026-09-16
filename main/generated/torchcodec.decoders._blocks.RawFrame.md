# RawFrame

*class*torchcodec.decoders._blocks.RawFrame(*handle: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *pts_seconds: [float](https://docs.python.org/3/builtins/functions.html#float)*, *duration_seconds: [float](https://docs.python.org/3/builtins/functions.html#float)*, *storage: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)[[source]](../_modules/torchcodec/decoders/_blocks/_frame.html#RawFrame)

One decoded video frame, exactly as the decoder produced it.

You cannot build one yourself: a [`VideoPacketDecoder`](torchcodec.decoders._blocks.VideoPacketDecoder.html#torchcodec.decoders._blocks.VideoPacketDecoder) creates
them. Use a [`ColorConverter`](torchcodec.decoders._blocks.ColorConverter.html#torchcodec.decoders._blocks.ColorConverter) to turn them into RGB
[`Frame`](torchcodec.Frame.html#torchcodec.Frame)s, or you can read and transform the raw
samples directly from `planes`:

```
for packet in demuxer:
 for raw_frame in packet_decoder.decode(packet):
 y, u, v = raw_frame.planes
 # For a 480x270 yuv420p frame, y is [270, 480] uint8, and the
 # chroma is subsampled: u and v are [135, 240] each.
 print(y.shape, u.shape, v.shape)
```

Nothing here has been converted. The samples are in the codec's own pixel
format (typically YUV), on the device that was passed to
[`VideoStream.make_decoder()`](torchcodec.decoders._blocks.VideoStream.html#torchcodec.decoders._blocks.VideoStream.make_decoder), and
`width`, `height` and `planes` are all pre-rotation:
`rotation_degrees` is what a [`ColorConverter`](torchcodec.decoders._blocks.ColorConverter.html#torchcodec.decoders._blocks.ColorConverter) applies for
you, and what you have to apply yourself if you convert `planes` on
your own.

Important

On CUDA, anything that reads the samples on a stream other than the one
the decoder ran on must call `record_stream()`, or the decoder may
overwrite them while those reads are still pending. A
[`ColorConverter`](torchcodec.decoders._blocks.ColorConverter.html#torchcodec.decoders._blocks.ColorConverter) does this for you.

Examples using `RawFrame`:

![](../_images/sphx_glr_raw_data_thumb.png)

[Raw frames and raw audio samples](../generated_examples/blocks/raw_data.html)

Raw frames and raw audio samples

*property*bit_depth*: [int](https://docs.python.org/3/builtins/functions.html#int)*

How many bits of each `planes` sample are meaningful.

In almost every case this is just the bit depth of the source: 8 for an
8-bit video, 10 for a 10-bit one. It is worth having because a plane's
dtype only tells you its storage width, `uint8` or `uint16`, while
this tells you the range of the values held in it. So it is what you
shift or scale by to reach a range of your own -
`y >> (frame.bit_depth - 8)` for 8 bits, or
`y / (2 ** frame.bit_depth - 1)` to normalise.

Two CUDA surface formats report more than their source: a 10-bit 4:4:4
source is uploaded as `yuv444p16le`, and a 12-bit source is tagged
`p016le` on FFmpeg < 6, which has no `p012le`. Both report 16 where
CPU decoding would report 10 and 12. Their samples are msb-aligned, so
they genuinely are 16-bit values with zeroed low bits, and the
arithmetic above still holds.

*property*color_range*: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

`"tv"` for limited range, `"pc"` for full range.

*property*colorspace*: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The FFmpeg colorspace name, e.g. `"bt709"`.

duration_seconds*: [float](https://docs.python.org/3/builtins/functions.html#float)*

How long this frame is displayed for, in seconds.

*property*height*: [int](https://docs.python.org/3/builtins/functions.html#int)*

The height of the decoded samples, before rotation.

*property*pix_fmt*: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The FFmpeg pixel-format name, e.g. `"yuv420p"`.

On CPU this is the source's own format. On CUDA it is always one of the
NVDEC surface formats: `"nv12"`, `"p010le"`, `"p012le"`,
`"p016le"`, `"yuv444p"` or `"yuv444p16le"`.

*property*planes*: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...]*

The decoder's own samples, as 2D tensor views.

There is exactly one tensor per *component*
of `pix_fmt`, in the order that format describes, of dtype
`uint8` or `uint16` depending on `bit_depth`. So `yuv420p`
and `nv12` both give three (`y, u, v = planes`), `yuva420p` four
(`y, u, v, a = planes`) and `gray` one (`(y,) = planes`).

They are always on the device that was passed to
[`VideoStream.make_decoder()`](torchcodec.decoders._blocks.VideoStream.html#torchcodec.decoders._blocks.VideoStream.make_decoder), including when a CUDA decoder has to
fall back to decoding on the CPU: it uploads those frames before handing
them out.

Only the luma and alpha components are `height` by `width`.
The chroma ones are subsampled by whatever `pix_fmt` says: half in
both directions for a 4:2:0 format, half the width for 4:2:2, full size
for 4:4:4 and for the RGB formats. Odd sizes round up, so the chroma of
a 4:2:0 frame 481 samples wide is 241 wide.

Note

None of these views is contiguous in general. FFmpeg pads each row
out to a line size of its own choosing, so even a luma plane is
usually strided, and the semi-planar formats (`nv12`, `p010le`,
and the other NVDEC surface formats) store U and V interleaved in a
single allocation, which forces those two to be strided views into
it.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - For the pixel formats that can't be viewed without a
 copy - sub-byte-packed, palettised and float ones - and for
 frames stored bottom-up. Check `pix_fmt` first if you are
 decoding something exotic.

pts_seconds*: [float](https://docs.python.org/3/builtins/functions.html#float)*

The [pts](../glossary.html#term-pts) of this frame, in seconds.

record_stream(*stream: [Stream](https://docs.pytorch.org/docs/stable/generated/torch.cuda.streams.Stream.html#torch.cuda.streams.Stream)*) → [None](https://docs.python.org/3/builtins/constants.html#None)[[source]](../_modules/torchcodec/decoders/_blocks/_frame.html#RawFrame.record_stream)

Tell the CUDA caching allocator that `stream` is still reading this
frame's samples.

**A CUDA consumer that reads the frame on a stream other than the one
the decoder ran on must call this**, right after queueing its reads.
Without it, the decoder's next frame can be handed the same buffer and
overwrite these samples while those reads are still pending.
[`ColorConverter`](torchcodec.decoders._blocks.ColorConverter.html#torchcodec.decoders._blocks.ColorConverter) does it for you, but you will have to call this
yourself if you consume `planes` directly on a different stream.

See [this post](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html) for
what the allocator is doing and why this is needed.

Parameters:

**stream** ([*torch.cuda.Stream*](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Stream_class.html#torch.cuda.Stream)) - The stream that is reading the samples.

*property*rotation_degrees*: [float](https://docs.python.org/3/builtins/functions.html#float)*

How many degrees counter-clockwise the frame has to be rotated to be
upright, or 0 if the container asks for no rotation.

This is *not* applied to `planes`. A [`ColorConverter`](torchcodec.decoders._blocks.ColorConverter.html#torchcodec.decoders._blocks.ColorConverter)
applies it, rounded to the nearest multiple of 90, to its output.

*property*width*: [int](https://docs.python.org/3/builtins/functions.html#int)*

The width of the decoded samples, before rotation.