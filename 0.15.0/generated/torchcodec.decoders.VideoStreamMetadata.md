# VideoStreamMetadata

*class*torchcodec.decoders.VideoStreamMetadata(*duration_seconds_from_header: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *begin_stream_seconds_from_header: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *bit_rate: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *codec: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*, *stream_index: [int](https://docs.python.org/3/library/functions.html#int)*, *duration_seconds: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *begin_stream_seconds: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *begin_stream_seconds_from_content: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *end_stream_seconds_from_content: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *width: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*, *height: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*, *num_frames_from_header: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*, *num_frames_from_content: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*, *average_fps_from_header: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *pixel_aspect_ratio: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction) | [None](https://docs.python.org/3/library/constants.html#None)*, *rotation: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *color_primaries: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*, *color_space: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*, *color_transfer_characteristic: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*, *pixel_format: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*, *end_stream_seconds: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *num_frames: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*, *average_fps: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*)[[source]](../_modules/torchcodec/_core/_metadata.html#VideoStreamMetadata)

Metadata of a single video stream.

Examples using `VideoStreamMetadata`:

![](../_images/sphx_glr_basic_example_thumb.png)

[Decoding a video with VideoDecoder](../generated_examples/decoding/basic_example.html)

Decoding a video with VideoDecoder

average_fps*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Average fps of the stream. If a [scan](../glossary.html#term-scan) was perfomed, this is
computed from the number of frames and the duration of the stream.
Otherwise we fall back to `average_fps_from_header`.

average_fps_from_header*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Averate fps of the stream, obtained from the header (float or None).
We recommend using the `average_fps` attribute instead.

begin_stream_seconds*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Beginning of the stream, in seconds (float). Conceptually, this
corresponds to the first frame's [pts](../glossary.html#term-pts). If a [scan](../glossary.html#term-scan) was performed
and `begin_stream_seconds_from_content` is not None, then it is returned.
Otherwise, this value is 0.

begin_stream_seconds_from_content*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Beginning of the stream, in seconds (float or None).
Conceptually, this corresponds to the first frame's [pts](../glossary.html#term-pts). It is only
computed when a [scan](../glossary.html#term-scan) is done as min(frame.pts) across all frames in
the stream. Usually, this is equal to 0.

begin_stream_seconds_from_header*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Beginning of the stream, in seconds, obtained from the header (float or
None). Usually, this is equal to 0.

bit_rate*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Bit rate of the stream, in seconds (float or None).

codec*: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

Codec (str or None).

color_primaries*: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

Color primaries as reported by FFmpeg. E.g. `"bt709"`, `"bt2020"`.

color_space*: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

Color space as reported by FFmpeg. E.g. `"bt709"`,
`"bt2020nc"`.

color_transfer_characteristic*: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

Color transfer characteristic as reported by FFmpeg
E.g. `"bt709"`, `"smpte2084"` (PQ), `"arib-std-b67"` (HLG).

duration_seconds*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Duration of the stream in seconds. We try to calculate the duration
from the actual frames if a [scan](../glossary.html#term-scan) was performed. Otherwise we
fall back to `duration_seconds_from_header`. If that value is also None,
we instead calculate the duration from `num_frames_from_header` and
`average_fps_from_header`. If all of those are unavailable, we fall back
to the container-level `duration_seconds_from_header`.

duration_seconds_from_header*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Duration of the stream, in seconds, obtained from the header (float or
None). This could be inaccurate.

end_stream_seconds*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

End of the stream, in seconds (float or None).
Conceptually, this corresponds to last_frame.pts + last_frame.duration.
If [scan](../glossary.html#term-scan) was performed and``end_stream_seconds_from_content`` is not None, then that value is
returned. Otherwise, returns `duration_seconds`.

end_stream_seconds_from_content*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

End of the stream, in seconds (float or None).
Conceptually, this corresponds to last_frame.pts + last_frame.duration. It
is only computed when a [scan](../glossary.html#term-scan) is done as max(frame.pts +
frame.duration) across all frames in the stream. Note that no frame is
played at this time value, so calling
[`get_frame_played_at()`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder.get_frame_played_at) with this
value would result in an error. Retrieving the last frame is best done by
simply indexing the [`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) object with
`[-1]`.

height*: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

Height of the frames (int or None).

num_frames*: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

Number of frames in the stream (int or None).
This corresponds to `num_frames_from_content` if a [scan](../glossary.html#term-scan) was made,
otherwise it corresponds to `num_frames_from_header`. If that value is also
None, the number of frames is calculated from the duration and the average fps.

num_frames_from_content*: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

Number of frames computed by TorchCodec by scanning the stream's
content (the scan doesn't involve decoding). This is more accurate
than `num_frames_from_header`. We recommend using the
`num_frames` attribute instead. (int or None).

num_frames_from_header*: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

Number of frames, from the stream's metadata. This is potentially
inaccurate. We recommend using the `num_frames` attribute instead.
(int or None).

pixel_aspect_ratio*: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction) | [None](https://docs.python.org/3/library/constants.html#None)*

Pixel Aspect Ratio (PAR), also known as Sample Aspect Ratio
(SAR -- not to be confused with Storage Aspect Ratio, also SAR),
is the ratio between the width and height of each pixel
(`fractions.Fraction` or None).

pixel_format*: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

The source pixel format of the video as reported by FFmpeg.
E.g. `'yuv420p'`, `'yuv444p'`, etc.

rotation*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Rotation angle in degrees (counter-clockwise rounded to the nearest
multiple of 90 degrees) from the display matrix metadata. This indicates
how the video should be rotated for correct display. TorchCodec automatically
applies this rotation during decoding, so the returned frames are in the
correct orientation (float or None).

Note

The `width` and
`height` attributes report
the **post-rotation** dimensions, i.e., the dimensions of frames as they
will be returned by TorchCodec's decoding methods. For videos with 90
or -90 degree rotation, this means width and height are swapped
compared to the raw encoded dimensions in the container.

stream_index*: [int](https://docs.python.org/3/library/functions.html#int)*

Index of the stream that this metadata refers to (int).

width*: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

Width of the frames (int or None).