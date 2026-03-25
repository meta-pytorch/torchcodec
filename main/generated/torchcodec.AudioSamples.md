# AudioSamples

*class*torchcodec.AudioSamples(*data: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *pts_seconds: [float](https://docs.python.org/3/library/functions.html#float)*, *duration_seconds: [float](https://docs.python.org/3/library/functions.html#float)*, *sample_rate: [int](https://docs.python.org/3/library/functions.html#int)*)[[source]](../_modules/torchcodec/_frame.html#AudioSamples)

Audio samples with associated metadata.

Examples using `AudioSamples`:

![](../_images/sphx_glr_audio_decoding_thumb.jpg)

[Decoding audio streams with AudioDecoder](../generated_examples/decoding/audio_decoding.html)

Decoding audio streams with AudioDecoder
![](../_images/sphx_glr_audio_encoding_thumb.jpg)

[Encoding audio samples with AudioEncoder](../generated_examples/encoding/audio_encoding.html)

Encoding audio samples with AudioEncoder

data*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

The sample data (`torch.Tensor` of float in [-1, 1], shape is `(num_channels, num_samples)`).

duration_seconds*: [float](https://docs.python.org/3/library/functions.html#float)*

The duration of the samples, in seconds.

pts_seconds*: [float](https://docs.python.org/3/library/functions.html#float)*

The [pts](../glossary.html#term-pts) of the first sample, in seconds.

sample_rate*: [int](https://docs.python.org/3/library/functions.html#int)*

The sample rate of the samples, in Hz.