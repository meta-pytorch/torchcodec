# RawAudioSamples

*class*torchcodec.decoders._blocks.RawAudioSamples(*data: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *sample_rate: [int](https://docs.python.org/3/builtins/functions.html#int)*, *pts_seconds: [float](https://docs.python.org/3/builtins/functions.html#float)*, *duration_seconds: [float](https://docs.python.org/3/builtins/functions.html#float)*, *_generation: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*)[[source]](../_modules/torchcodec/decoders/_blocks/_frame.html#RawAudioSamples)

One decoded audio frame's samples, exactly as the decoder produced them.

You cannot build one yourself: an [`AudioPacketDecoder`](torchcodec.decoders._blocks.AudioPacketDecoder.html#torchcodec.decoders._blocks.AudioPacketDecoder) creates them.
Use an [`AudioConverter`](torchcodec.decoders._blocks.AudioConverter.html#torchcodec.decoders._blocks.AudioConverter) to turn them into normalised float32
[`AudioSamples`](torchcodec.AudioSamples.html#torchcodec.AudioSamples), or read `data` directly:

```
for packet in demuxer:
 for raw_samples in audio_packet_decoder.decode(packet):
 print(raw_samples.data.shape) # e.g. [2, 1024]
 print(raw_samples.data.dtype) # e.g. float32, for an fltp source
 print(raw_samples.sample_rate) # e.g. 16000
```

Examples using `RawAudioSamples`:

![](../_images/sphx_glr_blocks_thumb.png)

[Blocks: build your own decoding pipeline](../generated_examples/decoding/blocks.html)

Blocks: build your own decoding pipeline

data*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Always a contiguous `[num_channels, num_samples]` tensor, whatever the
source's sample format. Planar and packed sources alike come out with that
same shape and layout (they are copied).

The dtype is whichever one holds the source's samples exactly: `uint8`
for `u8`, `int16` for `s16`, `int32` for `s32`, `int64` for
`s64`, `float32` for `flt` and `float64` for `dbl`. The integer
ones are *not* normalised to `[-1, 1]`; that is what an
[`AudioConverter`](torchcodec.decoders._blocks.AudioConverter.html#torchcodec.decoders._blocks.AudioConverter) does.

duration_seconds*: [float](https://docs.python.org/3/builtins/functions.html#float)*

How long these samples last, in seconds.

*property*num_channels*: [int](https://docs.python.org/3/builtins/functions.html#int)*

The number of channels, i.e. `data.shape[0]`.

*property*num_samples*: [int](https://docs.python.org/3/builtins/functions.html#int)*

The number of samples per channel, i.e. `data.shape[1]`.

pts_seconds*: [float](https://docs.python.org/3/builtins/functions.html#float)*

The [pts](../glossary.html#term-pts) of the first sample, in seconds.

sample_rate*: [int](https://docs.python.org/3/builtins/functions.html#int)*

The source's sample rate, in Hz.