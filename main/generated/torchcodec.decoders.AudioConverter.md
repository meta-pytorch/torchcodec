# AudioConverter

*class*torchcodec.decoders.AudioConverter(*sample_rate: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *num_channels: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)[[source]](../_modules/torchcodec/decoders/_blocks/_audio_converter.html#AudioConverter)

Turn [`RawAudioSamples`](torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples) into normalised float32
[`AudioSamples`](torchcodec.AudioSamples.html#torchcodec.AudioSamples), optionally resampling and remixing
channels.

Low-level API: for straightforward decoding, use
[`AudioDecoder`](torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) instead.

```
converter = AudioConverter(sample_rate=16_000)

for packet in demuxer:
 for raw_samples in packet_decoder.decode(packet):
 samples = converter.convert(raw_samples)
for raw_samples in packet_decoder.drain():
 samples = converter.convert(raw_samples)
samples = converter.drain()
```

Unlike a [`ColorConverter`](torchcodec.decoders.ColorConverter.html#torchcodec.decoders.ColorConverter), this object is a stateful stream
processor, and it is bound to an audio stream: feed it the stream's samples
in order. When resampling, it holds samples back between calls, so you won't
necessarily get the same number of samples out as you put in for a given
call to `convert()`.

Parameters:

- **sample_rate** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - The output sample rate. Defaults to the
source's own, i.e. no resampling.
- **num_channels** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - The output number of channels. Defaults
to the source's own.

Examples using `AudioConverter`:

![](../_images/sphx_glr_basics_thumb.png)

[Build your own decoding pipeline](../generated_examples/low_level/basics.html)

Build your own decoding pipeline

convert(*raw_samples: [RawAudioSamples](torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples)*) → [AudioSamples](torchcodec.AudioSamples.html#torchcodec.AudioSamples)[[source]](../_modules/torchcodec/decoders/_blocks/_audio_converter.html#AudioConverter.convert)

Convert one [`RawAudioSamples`](torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples) into normalised float32
[`AudioSamples`](torchcodec.AudioSamples.html#torchcodec.AudioSamples).

You may not get the same number of samples out as you put in, especially
if you are resampling.

Returns:

The converted samples, normalised float32 in `[-1, 1]`. When
resampling, fewer than were passed in, possibly none.

drain() → [AudioSamples](torchcodec.AudioSamples.html#torchcodec.AudioSamples)[[source]](../_modules/torchcodec/decoders/_blocks/_audio_converter.html#AudioConverter.drain)

Return the samples the resampler was still holding on to.

Empty unless you are resampling. This is not the codec's own buffer,
which [`AudioPacketDecoder.drain()`](torchcodec.decoders.AudioPacketDecoder.html#torchcodec.decoders.AudioPacketDecoder.drain) takes care of.

reset() → [None](https://docs.python.org/3/builtins/constants.html#None)[[source]](../_modules/torchcodec/decoders/_blocks/_audio_converter.html#AudioConverter.reset)

Drop the resampler's state and start over.

Needed after a [`Demuxer.seek()`](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer.seek), after `drain()`, and before
converting a different stream.