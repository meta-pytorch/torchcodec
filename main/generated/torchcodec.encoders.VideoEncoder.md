# VideoEncoder

*class*torchcodec.encoders.VideoEncoder(*frames: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *frame_rate: [float](https://docs.python.org/3/library/functions.html#float)*)[[source]](../_modules/torchcodec/encoders/_video_encoder.html#VideoEncoder)

A video encoder on CPU or CUDA..

Parameters:

- **frames** (`torch.Tensor`) - The frames to encode. This must be a 4D
tensor of shape `(N, C, H, W)` where N is the number of frames,
C is 3 channels (RGB), H is height, and W is width.
Values must be uint8 in the range `[0, 255]`.
The tensor can be on CPU or CUDA. The device of the tensor
determines which encoder is used (CPU or GPU).
- **frame_rate** ([*float*](https://docs.python.org/3/library/functions.html#float)) - The frame rate of the **input** `frames`. Also defines the encoded **output** frame rate.

Examples using `VideoEncoder`:

![](../_images/sphx_glr_video_encoding_thumb.jpg)

[Encoding video frames with VideoEncoder](../generated_examples/encoding/video_encoding.html)

Encoding video frames with VideoEncoder

to_file(*dest: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, ***, *codec: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *pixel_format: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *crf: [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *preset: [str](https://docs.python.org/3/library/stdtypes.html#str) | [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *extra_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torchcodec/encoders/_video_encoder.html#VideoEncoder.to_file)

Encode frames into a file.

Parameters:

- **dest** (str or `pathlib.Path`) - The path to the output file, e.g.
`video.mp4`. The extension of the file determines the video
container format.
- **codec** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**optional*) - The codec to use for encoding (e.g., "libx264",
"h264"). If not specified, the default codec
for the container format will be used.
See [Codec Selection](../generated_examples/encoding/video_encoding.html#codec-selection) for details.
- **pixel_format** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**optional*) - The pixel format for encoding (e.g.,
"yuv420p", "yuv444p"). If not specified, uses codec's default format.
Must be left as `None` when encoding CUDA tensors.
See [Pixel Format](../generated_examples/encoding/video_encoding.html#pixel-format) for details.
- **crf** ([*int*](https://docs.python.org/3/library/functions.html#int)*or*[*float*](https://docs.python.org/3/library/functions.html#float)*,**optional*) - Constant Rate Factor for encoding quality. Lower values
mean better quality. Valid range depends on the encoder (e.g. 0-51 for libx264).
Defaults to None (which will use encoder's default).
See [CRF (Constant Rate Factor)](../generated_examples/encoding/video_encoding.html#crf) for details.
- **preset** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - Encoder option that controls the tradeoff between
encoding encoding speed and compression (output size). Valid on the encoder (commonly
a string: "fast", "medium", "slow"). Defaults to None
(which will use encoder's default).
See [Preset](../generated_examples/encoding/video_encoding.html#preset) for details.
- **extra_options** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)*[*[*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**Any**]**,**optional*) - A dictionary of additional
encoder options to pass, e.g. `{"qp": 5, "tune": "film"}`.
See [Extra Options](../generated_examples/encoding/video_encoding.html#extra-options) for details.

to_file_like(*file_like*, *format: [str](https://docs.python.org/3/library/stdtypes.html#str)*, ***, *codec: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *pixel_format: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *crf: [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *preset: [str](https://docs.python.org/3/library/stdtypes.html#str) | [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *extra_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torchcodec/encoders/_video_encoder.html#VideoEncoder.to_file_like)

Encode frames into a file-like object.

Parameters:

- **file_like** - A file-like object that supports `write()` and
`seek()` methods, such as io.BytesIO(), an open file in binary
write mode, etc. Methods must have the following signature:
`write(data: bytes) -> int` and `seek(offset: int, whence:
int = 0) -> int`.
- **format** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) - The container format of the encoded frames, e.g. "mp4", "mov",
"mkv", "avi", "webm", "flv", etc.
- **codec** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**optional*) - The codec to use for encoding (e.g., "libx264",
"h264"). If not specified, the default codec
for the container format will be used.
See [Codec Selection](../generated_examples/encoding/video_encoding.html#codec-selection) for details.
- **pixel_format** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**optional*) - The pixel format for encoding (e.g.,
"yuv420p", "yuv444p"). If not specified, uses codec's default format.
Must be left as `None` when encoding CUDA tensors.
See [Pixel Format](../generated_examples/encoding/video_encoding.html#pixel-format) for details.
- **crf** ([*int*](https://docs.python.org/3/library/functions.html#int)*or*[*float*](https://docs.python.org/3/library/functions.html#float)*,**optional*) - Constant Rate Factor for encoding quality. Lower values
mean better quality. Valid range depends on the encoder (e.g. 0-51 for libx264).
Defaults to None (which will use encoder's default).
See [CRF (Constant Rate Factor)](../generated_examples/encoding/video_encoding.html#crf) for details.
- **preset** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - Encoder option that controls the tradeoff between
encoding encoding speed and compression (output size). Valid on the encoder (commonly
a string: "fast", "medium", "slow"). Defaults to None
(which will use encoder's default).
See [Preset](../generated_examples/encoding/video_encoding.html#preset) for details.
- **extra_options** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)*[*[*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**Any**]**,**optional*) - A dictionary of additional
encoder options to pass, e.g. `{"qp": 5, "tune": "film"}`.
See [Extra Options](../generated_examples/encoding/video_encoding.html#extra-options) for details.

to_tensor(*format: [str](https://docs.python.org/3/library/stdtypes.html#str)*, ***, *codec: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *pixel_format: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *crf: [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *preset: [str](https://docs.python.org/3/library/stdtypes.html#str) | [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *extra_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../_modules/torchcodec/encoders/_video_encoder.html#VideoEncoder.to_tensor)

Encode frames into raw bytes, as a 1D uint8 Tensor.

Parameters:

- **format** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) - The container format of the encoded frames, e.g. "mp4", "mov",
"mkv", "avi", "webm", "flv", etc.
- **codec** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**optional*) - The codec to use for encoding (e.g., "libx264",
"h264"). If not specified, the default codec
for the container format will be used.
See [Codec Selection](../generated_examples/encoding/video_encoding.html#codec-selection) for details.
- **pixel_format** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**optional*) - The pixel format to encode frames into (e.g.,
"yuv420p", "yuv444p"). If not specified, uses codec's default format.
Must be left as `None` when encoding CUDA tensors.
See [Pixel Format](../generated_examples/encoding/video_encoding.html#pixel-format) for details.
- **crf** ([*int*](https://docs.python.org/3/library/functions.html#int)*or*[*float*](https://docs.python.org/3/library/functions.html#float)*,**optional*) - Constant Rate Factor for encoding quality. Lower values
mean better quality. Valid range depends on the encoder (e.g. 0-51 for libx264).
Defaults to None (which will use encoder's default).
See [CRF (Constant Rate Factor)](../generated_examples/encoding/video_encoding.html#crf) for details.
- **preset** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - Encoder option that controls the tradeoff between
encoding encoding speed and compression (output size). Valid on the encoder (commonly
a string: "fast", "medium", "slow"). Defaults to None
(which will use encoder's default).
See [Preset](../generated_examples/encoding/video_encoding.html#preset) for details.
- **extra_options** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)*[*[*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**Any**]**,**optional*) - A dictionary of additional
encoder options to pass, e.g. `{"qp": 5, "tune": "film"}`.
See [Extra Options](../generated_examples/encoding/video_encoding.html#extra-options) for details.

Returns:

The raw encoded bytes as 1D uint8 Tensor on CPU regardless of the device of the input frames.

Return type:

Tensor