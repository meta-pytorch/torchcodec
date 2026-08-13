# WavDecoder

*class*torchcodec.decoders.WavDecoder(*source: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [RawIOBase](https://docs.python.org/3/library/io.html#io.RawIOBase) | BufferedReader | [bytes](https://docs.python.org/3/library/stdtypes.html#bytes) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*)[[source]](../_modules/torchcodec/decoders/_wav_decoder.html#WavDecoder)

A fast decoder for WAV audio files.

This is a lightweight, high-performance alternative to
[`AudioDecoder`](torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) that is specialized for WAV
files. See [TorchCodec Performance Tips and Best Practices](../generated_examples/decoding/performance_tips.html#sphx-glr-generated-examples-decoding-performance-tips-py)
for more details.

Unlike [`AudioDecoder`](torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder), this decoder does not
support resampling (`sample_rate` parameter) or channel remixing
(`num_channels` parameter). If you need those features, use
[`AudioDecoder`](torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder).

Returned samples are float samples normalized in [-1, 1].

Parameters:

- **(****str** (*source*) - 

object): The source of the audio:

- If `str`: a path to a local WAV file.
- If `Pathlib.path`: a path to a local WAV file.
- If `bytes` object or `torch.Tensor`: the raw WAV data.
- If file-like object: we read audio data from the object on
demand. The object must expose the methods `read(self, size:
int) -> bytes` and `seek(self, offset: int, whence: int) ->
int`.
- **Pathlib.path** - 

object): The source of the audio:

- If `str`: a path to a local WAV file.
- If `Pathlib.path`: a path to a local WAV file.
- If `bytes` object or `torch.Tensor`: the raw WAV data.
- If file-like object: we read audio data from the object on
demand. The object must expose the methods `read(self, size:
int) -> bytes` and `seek(self, offset: int, whence: int) ->
int`.
- **bytes** - 

object): The source of the audio:

- If `str`: a path to a local WAV file.
- If `Pathlib.path`: a path to a local WAV file.
- If `bytes` object or `torch.Tensor`: the raw WAV data.
- If file-like object: we read audio data from the object on
demand. The object must expose the methods `read(self, size:
int) -> bytes` and `seek(self, offset: int, whence: int) ->
int`.
- **file-like** (*torch.Tensor or*) - 

object): The source of the audio:

- If `str`: a path to a local WAV file.
- If `Pathlib.path`: a path to a local WAV file.
- If `bytes` object or `torch.Tensor`: the raw WAV data.
- If file-like object: we read audio data from the object on
demand. The object must expose the methods `read(self, size:
int) -> bytes` and `seek(self, offset: int, whence: int) ->
int`.

Variables:

- **metadata** ([*AudioStreamMetadata*](torchcodec.decoders.AudioStreamMetadata.html#torchcodec.decoders.AudioStreamMetadata)) - Metadata of the audio stream.
- **stream_index** ([*int*](https://docs.python.org/3/library/functions.html#int)) - The stream index. Always 0 for WAV files.

Examples using `WavDecoder`:

![](../_images/sphx_glr_audio_decoding_thumb.jpg)

[Decoding audio streams with AudioDecoder](../generated_examples/decoding/audio_decoding.html)

Decoding audio streams with AudioDecoder
![](../_images/sphx_glr_performance_tips_thumb.jpg)

[TorchCodec Performance Tips and Best Practices](../generated_examples/decoding/performance_tips.html)

TorchCodec Performance Tips and Best Practices

get_all_samples() → [AudioSamples](torchcodec.AudioSamples.html#torchcodec.AudioSamples)[[source]](../_modules/torchcodec/decoders/_wav_decoder.html#WavDecoder.get_all_samples)

Returns all the audio samples from the source.

To decode samples in a specific range, use
`get_samples_played_in_range()`.

Returns:

The samples within the file.

Return type:

[AudioSamples](torchcodec.AudioSamples.html#torchcodec.AudioSamples)

get_samples_played_in_range(*start_seconds: [float](https://docs.python.org/3/library/functions.html#float) = 0.0*, *stop_seconds: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [AudioSamples](torchcodec.AudioSamples.html#torchcodec.AudioSamples)[[source]](../_modules/torchcodec/decoders/_wav_decoder.html#WavDecoder.get_samples_played_in_range)

Returns audio samples in the given range.

Samples are in the half open range [start_seconds, stop_seconds).

To decode all the samples from beginning to end, you can call this
method while leaving `start_seconds` and `stop_seconds` to their
default values, or use
`get_all_samples()` as a more
convenient alias.

Parameters:

- **start_seconds** ([*float*](https://docs.python.org/3/library/functions.html#float)) - Time, in seconds, of the start of the
range. Default: 0.
- **stop_seconds** ([*float*](https://docs.python.org/3/library/functions.html#float)*or**None*) - Time, in seconds, of the end of the
range. As a half open range, the end is excluded. Default: None,
which decodes samples until the end.

Returns:

The samples within the specified range.

Return type:

[AudioSamples](torchcodec.AudioSamples.html#torchcodec.AudioSamples)