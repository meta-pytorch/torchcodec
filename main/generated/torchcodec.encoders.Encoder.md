# Encoder

*class*torchcodec.encoders.Encoder[[source]](../_modules/torchcodec/encoders/_multi_stream_encoder.html#Encoder)

A multi-stream encoder for encoding video and/or audio into a file or file-like object.

Unlike [`VideoEncoder`](torchcodec.encoders.VideoEncoder.html#torchcodec.encoders.VideoEncoder) and [`AudioEncoder`](torchcodec.encoders.AudioEncoder.html#torchcodec.encoders.AudioEncoder) which encode a
single stream in one shot, `Encoder` supports multiple streams and
incremental (streaming) encoding. Frames and samples can be added
progressively, which is useful when data is generated on-the-fly or when
encoding both audio and video into the same container.

Use `add_video()` and `add_audio()` to configure output streams,
then open an output destination with `open_file()` or
`open_file_like()`, feed data via the returned stream objects, and
finally call `close()` (or use the encoder as a context manager).

Example

```
with Encoder() as encoder:
 video_stream = encoder.add_video(height=256, width=256, frame_rate=30)
 audio_stream = encoder.add_audio(sample_rate=16000, num_channels=1)
 encoder.open_file("output.mp4")
 video_stream.add_frames(frames_tensor)
 audio_stream.add_samples(samples_tensor)
```

add_audio(***, *sample_rate: [int](https://docs.python.org/3/library/functions.html#int)*, *num_channels: [int](https://docs.python.org/3/library/functions.html#int)*, *bit_rate: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *out_num_channels: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *out_sample_rate: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [AudioStream](torchcodec.encoders.AudioStream.html#torchcodec.encoders.AudioStream)[[source]](../_modules/torchcodec/encoders/_multi_stream_encoder.html#Encoder.add_audio)

Add an audio stream to the encoder.

Must be called before `open_file()` or `open_file_like()`.

Parameters:

- **sample_rate** ([*int*](https://docs.python.org/3/library/functions.html#int)) - The sample rate of the **input** samples.
- **num_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)) - The number of channels of the **input** samples.
- **bit_rate** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The output bit rate. Encoders typically
support a finite set of bit rate values, so `bit_rate` will be
matched to one of those supported values. The default is chosen
by FFmpeg.
- **out_num_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The number of channels of the
encoded output. By default, the input `num_channels` is used.
- **out_sample_rate** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The sample rate of the encoded
output. By default, the input `sample_rate` is used.

Returns:

An audio stream object. Use its [`add_samples()`](torchcodec.encoders.AudioStream.html#torchcodec.encoders.AudioStream.add_samples)
method to feed samples into the stream.

add_video(***, *height: [int](https://docs.python.org/3/library/functions.html#int)*, *width: [int](https://docs.python.org/3/library/functions.html#int)*, *frame_rate: [float](https://docs.python.org/3/library/functions.html#float)*, *device: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'cpu'*, *codec: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *pixel_format: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *crf: [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *preset: [str](https://docs.python.org/3/library/stdtypes.html#str) | [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *extra_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [VideoStream](torchcodec.encoders.VideoStream.html#torchcodec.encoders.VideoStream)[[source]](../_modules/torchcodec/encoders/_multi_stream_encoder.html#Encoder.add_video)

Add a video stream to the encoder.

Must be called before `open_file()` or `open_file_like()`.

Parameters:

- **height** ([*int*](https://docs.python.org/3/library/functions.html#int)) - The height of the **input** video frames.
- **width** ([*int*](https://docs.python.org/3/library/functions.html#int)) - The width of the **input** video frames.
- **frame_rate** ([*float*](https://docs.python.org/3/library/functions.html#float)) - The frame rate of the **input** video frames.
Also defines the encoded **output** frame rate.
- **device** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**optional*) - The device to use for encoding, e.g.
`"cpu"` or `"cuda"`. Default: `"cpu"`.
- **codec** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**optional*) - The codec to use for encoding (e.g.,
`"libx264"`). If not specified, the default codec for the
container format will be used.
See [Codec Selection](../generated_examples/encoding/video_encoding.html#codec-selection) for details.
- **pixel_format** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**optional*) - The pixel format for encoding (e.g.,
`"yuv420p"`). If not specified, uses codec's default format.
Must be left as `None` when encoding on CUDA.
See [Pixel Format](../generated_examples/encoding/video_encoding.html#pixel-format) for details.
- **crf** ([*int*](https://docs.python.org/3/library/functions.html#int)*or*[*float*](https://docs.python.org/3/library/functions.html#float)*,**optional*) - Constant Rate Factor for encoding
quality. Lower values mean better quality. Valid range depends
on the encoder (e.g. 0-51 for libx264). Defaults to None (which
will use encoder's default). See [CRF (Constant Rate Factor)](../generated_examples/encoding/video_encoding.html#crf) for details.
- **preset** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - Encoder option that controls the
tradeoff between encoding speed and compression (output size).
Commonly a string: `"fast"`, `"medium"`, `"slow"`.
Defaults to None (which will use encoder's default).
See [Preset](../generated_examples/encoding/video_encoding.html#preset) for details.
- **extra_options** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)*[*[*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**Any**]**,**optional*) - A dictionary of additional
encoder options to pass, e.g. `{"qp": 5, "tune": "film"}`.
See [Extra Options](../generated_examples/encoding/video_encoding.html#extra-options) for details.

Returns:

A video stream object. Use its [`add_frames()`](torchcodec.encoders.VideoStream.html#torchcodec.encoders.VideoStream.add_frames)
method to feed frames into the stream.

close() → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torchcodec/encoders/_multi_stream_encoder.html#Encoder.close)

Flush all remaining data and close the encoder.

This must be called when encoding is complete to ensure all buffered
data is written. Using the encoder as a context manager (`with`
statement) calls this automatically.

open_file(*dest: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*) → Encoder[[source]](../_modules/torchcodec/encoders/_multi_stream_encoder.html#Encoder.open_file)

Open a file for writing the encoded output.

Must be called after all streams have been added via `add_video()`
and/or `add_audio()`. The file extension determines the container
format (e.g. `.mp4`, `.mkv`).

Parameters:

**dest** (str or `pathlib.Path`) - The path to the output file.

Returns:

Returns `self` for method chaining.

Return type:

Encoder

open_file_like(*dest*, ***, *format: [str](https://docs.python.org/3/library/stdtypes.html#str)*) → Encoder[[source]](../_modules/torchcodec/encoders/_multi_stream_encoder.html#Encoder.open_file_like)

Open a file-like object for writing the encoded output.

Must be called after all streams have been added via `add_video()`
and/or `add_audio()`.

Parameters:

- **dest** - A file-like object that supports `write()` and `seek()`
methods, such as `io.BytesIO()`, an open file in binary write
mode, etc. Methods must have the following signature:
`write(data: bytes) -> int` and `seek(offset: int, whence:
int = 0) -> int`.
- **format** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) - The container format of the encoded output, e.g.
`"mp4"`, `"mov"`, `"mkv"`, `"avi"`, `"webm"`, etc.

Returns:

Returns `self` for method chaining.

Return type:

Encoder