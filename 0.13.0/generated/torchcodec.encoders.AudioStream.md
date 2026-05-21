# AudioStream

*class*torchcodec.encoders.AudioStream(*encoder_tensor: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *stream_index: [int](https://docs.python.org/3/library/functions.html#int)*)[[source]](../_modules/torchcodec/encoders/_multi_stream_encoder.html#AudioStream)

An audio stream within an [`Encoder`](torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder).

Returned by [`Encoder.add_audio()`](torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder.add_audio). Use `add_samples()` to feed
audio samples into this stream.

Examples using `AudioStream`:

![](../_images/sphx_glr_multi_stream_encoding_thumb.jpg)

[Encoding audio and video streams with the Encoder](../generated_examples/encoding/multi_stream_encoding.html)

Encoding audio and video streams with the Encoder

add_samples(*samples: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torchcodec/encoders/_multi_stream_encoder.html#AudioStream.add_samples)

Add audio samples to this stream.

Parameters:

**samples** (`torch.Tensor`) - The samples to encode. This must be a
2D tensor of shape `(num_channels, num_samples)`. Values must
be float values in `[-1, 1]`. The number of channels must
match the `num_channels` passed to [`Encoder.add_audio()`](torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder.add_audio).