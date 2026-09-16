# VideoStreamHeaderMetadata

*class*torchcodec.decoders._blocks.VideoStreamHeaderMetadata(*duration_seconds_from_header: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *begin_stream_seconds_from_header: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *bit_rate: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *codec: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *stream_index: [int](https://docs.python.org/3/builtins/functions.html#int)*, *media_type: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*, *width: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *height: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *num_frames_from_header: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *average_fps_from_header: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *pixel_aspect_ratio: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *rotation: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *color_primaries: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *color_space: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *color_transfer_characteristic: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *pixel_format: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*)[[source]](../_modules/torchcodec/_core/_metadata.html#VideoStreamHeaderMetadata)

Metadata of a single video stream, as reported by the container header.

Examples using `VideoStreamHeaderMetadata`:

![](../_images/sphx_glr_basics_thumb.png)

[Blocks: build your own decoding pipeline](../generated_examples/blocks/basics.html)

Blocks: build your own decoding pipeline

average_fps_from_header*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Averate fps of the stream, obtained from the header (float or None).
We recommend using the `average_fps` attribute instead.

begin_stream_seconds_from_header*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Beginning of the stream, in seconds, obtained from the header (float or
None). Usually, this is equal to 0.

bit_rate*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Bit rate of the stream, in seconds (float or None).

codec*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Codec (str or None).

color_primaries*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Color primaries as reported by FFmpeg. E.g. `"bt709"`, `"bt2020"`.

color_space*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Color space as reported by FFmpeg. E.g. `"bt709"`,
`"bt2020nc"`.

color_transfer_characteristic*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Color transfer characteristic as reported by FFmpeg
E.g. `"bt709"`, `"smpte2084"` (PQ), `"arib-std-b67"` (HLG).

duration_seconds_from_header*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Duration of the stream, in seconds, obtained from the header (float or
None). This could be inaccurate.

height*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Height of the frames (int or None).

media_type*: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Type of media the stream carries (str). One of `"video"`, `"audio"`,
`"subtitle"`, `"data"`, `"attachment"` or `"unknown"`.

num_frames_from_header*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Number of frames, from the stream's metadata. This is potentially
inaccurate.
(int or None).

pixel_aspect_ratio*: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Pixel Aspect Ratio (PAR), also known as Sample Aspect Ratio
(SAR -- not to be confused with Storage Aspect Ratio, also SAR),
is the ratio between the width and height of each pixel
(`fractions.Fraction` or None).

pixel_format*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The source pixel format of the video as reported by FFmpeg.
E.g. `'yuv420p'`, `'yuv444p'`, etc.

rotation*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Rotation angle in degrees (counter-clockwise rounded to the nearest
multiple of 90 degrees) from the display matrix metadata. This indicates
how the video should be rotated for correct display. TorchCodec automatically
applies this rotation during decoding, so the returned frames are in the
correct orientation (float or None).

Note

The [`width`](torchcodec.decoders.VideoStreamMetadata.html#torchcodec.decoders.VideoStreamMetadata.width) and
[`height`](torchcodec.decoders.VideoStreamMetadata.html#torchcodec.decoders.VideoStreamMetadata.height) attributes report
the **post-rotation** dimensions, i.e., the dimensions of frames as they
will be returned by TorchCodec's decoding methods. For videos with 90
or -90 degree rotation, this means width and height are swapped
compared to the raw encoded dimensions in the container.

stream_index*: [int](https://docs.python.org/3/builtins/functions.html#int)*

Index of the stream that this metadata refers to (int).

width*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Width of the frames (int or None).