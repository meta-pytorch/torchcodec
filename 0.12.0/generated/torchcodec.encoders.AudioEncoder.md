# AudioEncoder

*class*torchcodec.encoders.AudioEncoder(*samples: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *sample_rate: [int](https://docs.python.org/3/library/functions.html#int)*)[[source]](../_modules/torchcodec/encoders/_audio_encoder.html#AudioEncoder)

An audio encoder.

Parameters:

- **samples** (`torch.Tensor`) - The samples to encode. This must be a 2D
tensor of shape `(num_channels, num_samples)`, or a 1D tensor in
which case `num_channels = 1` is assumed. Values must be float
values in `[-1, 1]`.
- **sample_rate** ([*int*](https://docs.python.org/3/library/functions.html#int)) - The sample rate of the **input** `samples`. The
sample rate of the encoded output can be specified using the
encoding methods (`to_file`, etc.).

Examples using `AudioEncoder`:

![](../_images/sphx_glr_audio_encoding_thumb.jpg)

[Encoding audio samples with AudioEncoder](../generated_examples/encoding/audio_encoding.html)

Encoding audio samples with AudioEncoder

to_file(*dest: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, ***, *bit_rate: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *num_channels: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *sample_rate: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torchcodec/encoders/_audio_encoder.html#AudioEncoder.to_file)

Encode samples into a file.

Parameters:

- **dest** (str or `pathlib.Path`) - The path to the output file, e.g.
`audio.mp3`. The extension of the file determines the audio
format and container.
- **bit_rate** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The output bit rate. Encoders typically
support a finite set of bit rate values, so `bit_rate` will be
matched to one of those supported values. The default is chosen
by FFmpeg.
- **num_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The number of channels of the encoded
output samples. By default, the number of channels of the input
`samples` is used.
- **sample_rate** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The sample rate of the encoded output.
By default, the sample rate of the input `samples` is used.

to_file_like(*file_like*, *format: [str](https://docs.python.org/3/library/stdtypes.html#str)*, ***, *bit_rate: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *num_channels: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *sample_rate: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torchcodec/encoders/_audio_encoder.html#AudioEncoder.to_file_like)

Encode samples into a file-like object.

Parameters:

- **file_like** - A file-like object that supports `write()` and
`seek()` methods, such as io.BytesIO(), an open file in binary
write mode, etc. Methods must have the following signature:
`write(data: bytes) -> int` and `seek(offset: int, whence:
int = 0) -> int`.
- **format** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) - The format of the encoded samples, e.g. "mp3", "wav"
or "flac".
- **bit_rate** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The output bit rate. Encoders typically
support a finite set of bit rate values, so `bit_rate` will be
matched to one of those supported values. The default is chosen
by FFmpeg.
- **num_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The number of channels of the encoded
output samples. By default, the number of channels of the input
`samples` is used.
- **sample_rate** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The sample rate of the encoded output.
By default, the sample rate of the input `samples` is used.

to_tensor(*format: [str](https://docs.python.org/3/library/stdtypes.html#str)*, ***, *bit_rate: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *num_channels: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *sample_rate: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../_modules/torchcodec/encoders/_audio_encoder.html#AudioEncoder.to_tensor)

Encode samples into raw bytes, as a 1D uint8 Tensor.

Parameters:

- **format** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) - The format of the encoded samples, e.g. "mp3", "wav"
or "flac".
- **bit_rate** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The output bit rate. Encoders typically
support a finite set of bit rate values, so `bit_rate` will be
matched to one of those supported values. The default is chosen
by FFmpeg.
- **num_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The number of channels of the encoded
output samples. By default, the number of channels of the input
`samples` is used.
- **sample_rate** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The sample rate of the encoded output.
By default, the sample rate of the input `samples` is used.

Returns:

The raw encoded bytes as 1D uint8 Tensor.

Return type:

Tensor