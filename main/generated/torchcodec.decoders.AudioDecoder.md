# AudioDecoder

*class*torchcodec.decoders.AudioDecoder(*source: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [RawIOBase](https://docs.python.org/3/library/io.html#io.RawIOBase) | BufferedReader | [bytes](https://docs.python.org/3/library/stdtypes.html#bytes) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *stream_index: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *sample_rate: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *num_channels: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*)[[source]](../_modules/torchcodec/decoders/_audio_decoder.html#AudioDecoder)

A single-stream audio decoder.

This can be used to decode audio from pure audio files (e.g. mp3, wav,
etc.), or from videos that contain audio streams (e.g. mp4 videos).

Returned samples are float samples normalized in [-1, 1]

Parameters:

- **(****str** (*source*) - 

object): The source of the video or audio:

- If `str`: a local path or a URL to a video or audio file.
- If `Pathlib.path`: a path to a local video or audio file.
- If `bytes` object or `torch.Tensor`: the raw encoded audio data.
- If file-like object: we read video data from the object on demand. The object must
expose the methods read(self, size: int) -> bytes and
seek(self, offset: int, whence: int) -> int. Read more in:
[Streaming data through file-like support](../generated_examples/decoding/file_like.html#sphx-glr-generated-examples-decoding-file-like-py).
- **Pathlib.path** - 

object): The source of the video or audio:

- If `str`: a local path or a URL to a video or audio file.
- If `Pathlib.path`: a path to a local video or audio file.
- If `bytes` object or `torch.Tensor`: the raw encoded audio data.
- If file-like object: we read video data from the object on demand. The object must
expose the methods read(self, size: int) -> bytes and
seek(self, offset: int, whence: int) -> int. Read more in:
[Streaming data through file-like support](../generated_examples/decoding/file_like.html#sphx-glr-generated-examples-decoding-file-like-py).
- **bytes** - 

object): The source of the video or audio:

- If `str`: a local path or a URL to a video or audio file.
- If `Pathlib.path`: a path to a local video or audio file.
- If `bytes` object or `torch.Tensor`: the raw encoded audio data.
- If file-like object: we read video data from the object on demand. The object must
expose the methods read(self, size: int) -> bytes and
seek(self, offset: int, whence: int) -> int. Read more in:
[Streaming data through file-like support](../generated_examples/decoding/file_like.html#sphx-glr-generated-examples-decoding-file-like-py).
- **file-like** (*torch.Tensor or*) - 

object): The source of the video or audio:

- If `str`: a local path or a URL to a video or audio file.
- If `Pathlib.path`: a path to a local video or audio file.
- If `bytes` object or `torch.Tensor`: the raw encoded audio data.
- If file-like object: we read video data from the object on demand. The object must
expose the methods read(self, size: int) -> bytes and
seek(self, offset: int, whence: int) -> int. Read more in:
[Streaming data through file-like support](../generated_examples/decoding/file_like.html#sphx-glr-generated-examples-decoding-file-like-py).
- **stream_index** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - Specifies which stream in the file to decode samples from.
Note that this index is absolute across all media types. If left unspecified, then
the [best stream](../glossary.html#term-best-stream) is used.
- **sample_rate** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The desired output sample rate of the decoded samples.
By default, the sample rate of the source is used.
- **num_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The desired number of channels of the decoded samples.
By default, the number of channels of the source is used.

Variables:

- **metadata** ([*AudioStreamMetadata*](torchcodec.decoders.AudioStreamMetadata.html#torchcodec.decoders.AudioStreamMetadata)) - Metadata of the audio stream.
- **stream_index** ([*int*](https://docs.python.org/3/library/functions.html#int)) - The stream index that this decoder is retrieving samples from. If a
stream index was provided at initialization, this is the same value. If it was left
unspecified, this is the [best stream](../glossary.html#term-best-stream).

Examples using `AudioDecoder`:

![](../_images/sphx_glr_audio_decoding_thumb.jpg)

[Decoding audio streams with AudioDecoder](../generated_examples/decoding/audio_decoding.html)

Decoding audio streams with AudioDecoder
![](../_images/sphx_glr_file_like_thumb.jpg)

[Streaming data through file-like support](../generated_examples/decoding/file_like.html)

Streaming data through file-like support
![](../_images/sphx_glr_audio_encoding_thumb.jpg)

[Encoding audio samples with AudioEncoder](../generated_examples/encoding/audio_encoding.html)

Encoding audio samples with AudioEncoder

get_all_samples() → [AudioSamples](torchcodec.AudioSamples.html#torchcodec.AudioSamples)[[source]](../_modules/torchcodec/decoders/_audio_decoder.html#AudioDecoder.get_all_samples)

Returns all the audio samples from the source.

To decode samples in a specific range, use
`get_samples_played_in_range()`.

Returns:

The samples within the file.

Return type:

[AudioSamples](torchcodec.AudioSamples.html#torchcodec.AudioSamples)

get_samples_played_in_range(*start_seconds: [float](https://docs.python.org/3/library/functions.html#float) = 0.0*, *stop_seconds: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [AudioSamples](torchcodec.AudioSamples.html#torchcodec.AudioSamples)[[source]](../_modules/torchcodec/decoders/_audio_decoder.html#AudioDecoder.get_samples_played_in_range)

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