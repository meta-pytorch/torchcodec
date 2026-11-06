# VideoCodecParameters

*class*torchcodec.decoders.VideoCodecParameters(*codec: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*, ***, *time_base: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction)*, *extradata: [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *codec_tag: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *profile: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *level: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *bit_rate: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *bits_per_coded_sample: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *bits_per_raw_sample: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *coded_side_data: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *width: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *height: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *pixel_format: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *sample_aspect_ratio: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *framerate: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *r_frame_rate: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *field_order: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *color_range: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *color_primaries: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *color_transfer_characteristic: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *color_space: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *chroma_location: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *video_delay: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)[[source]](../_modules/torchcodec/decoders/_blocks/_codec_parameters.html#VideoCodecParameters)

What a [`VideoPacketDecoder`](torchcodec.decoders.VideoPacketDecoder.html#torchcodec.decoders.VideoPacketDecoder) needs to know to decode the packets
of a video stream.

If you have a [`Demuxer`](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer), [`VideoStream.codec_parameters`](torchcodec.decoders.VideoStream.html#torchcodec.decoders.VideoStream.codec_parameters) gives
you the parameters of a stream. They can be pickled, e.g. to build the
decoder in another process. To decode packets that don't come from a
`Demuxer`, build them yourself:

```
codec_parameters = VideoCodecParameters(
 "h264", time_base=Fraction(1, 90_000), extradata=extradata
)
decoder = VideoPacketDecoder(codec_parameters)
```

The fields mirror those of FFmpeg's [AVCodecParameters](https://ffmpeg.org/doxygen/trunk/structAVCodecParameters.html), with the
same names and meanings. FFmpeg's names stand for its enums, e.g.
`pixel_format="yuv420p"`.

Note

Which fields you need depends on the codec and device: printing the parameters a
`Demuxer` reports for a similar file shows what FFmpeg uses.

To decode on CUDA, set `width`, `height` and `pixel_format`:
depending on the codec, the frames may otherwise be silently decoded
on the CPU. Also set `r_frame_rate`, or the frames' durations are 0.

Note

The parameters are meant to be passed to a decoder or stored, not read
as metadata: use [`VideoStream.metadata`](torchcodec.decoders.VideoStream.html#torchcodec.decoders.VideoStream.metadata) for that. The values a
`Demuxer` reports are FFmpeg's, and can change across FFmpeg and
TorchCodec versions. The fields' names and meanings are stable.

Examples using `VideoCodecParameters`:

![](../_images/sphx_glr_basics_thumb.jpg)

[Build your own decoding pipeline](../generated_examples/low_level/basics.html)

Build your own decoding pipeline
![](../_images/sphx_glr_demuxer_free_thumb.jpg)

[Decoding packets that don't come from a Demuxer](../generated_examples/low_level/demuxer_free.html)

Decoding packets that don't come from a Demuxer

bit_rate*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The average bit rate, in bits per second.

bits_per_coded_sample*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The number of bits per sample in the compressed data, for the codecs
that need it (mostly uncompressed and PCM codecs).

bits_per_raw_sample*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The number of bits per sample in the decoded data, when it's not implied
by the pixel or sample format, e.g. 10 for 10-bit video.

chroma_location*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

FFmpeg's name of the chroma samples' location, e.g. `"left"`.

codec*: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

FFmpeg's name of the codec, e.g. `"h264"`, `"hevc"` or `"aac"`.
`ffmpeg -codecs` lists them.

codec_tag*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The container's identifier of the codec, e.g. a FourCC in AVI or MP4.
A few codecs read it to tell variants apart.

coded_side_data*: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes)] | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

Extra data about the stream, keyed by FFmpeg's names for its types, e.g.
`"Display Matrix"` for video or `"Replay Gain"` for audio.

If a video stream has a display matrix, it is under `"Display Matrix"`
on every FFmpeg version.

color_primaries*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

FFmpeg's name of the color primaries, e.g. `"bt709"`.

color_range*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

FFmpeg's name of the color range, e.g. `"tv"` (limited) or `"pc"`
(full).

color_space*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

FFmpeg's name of the color space, i.e. the YUV matrix, e.g.
`"bt709"`.

color_transfer_characteristic*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

FFmpeg's name of the transfer characteristic, e.g. `"bt709"` or
`"smpte2084"`.

extradata*: [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The codec's setup data that isn't in the packets, e.g. the `avcC` box
of H.264 in MP4. Not all codecs and streams need it.

field_order*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

`"progressive"` or, for interlaced video, which field comes first:
`"tt"`, `"bb"`, `"tb"` or `"bt"`.

framerate*: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The frame rate, in frames per second, as the codec reports it. Setting
it requires FFmpeg 6.1 or above.

height*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The height of the frames, in pixels.

level*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The codec's level.

pixel_format*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

FFmpeg's name of the decoded frames' pixel format, e.g. `"yuv420p"`.
`ffmpeg -pix_fmts` lists them.

profile*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The codec's profile, as one of FFmpeg's `AV_PROFILE_*` values.

r_frame_rate*: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The frame rate, in frames per second, as the container reports it. Not
part of FFmpeg's `AVCodecParameters`, but of its `AVStream`.

sample_aspect_ratio*: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The width of a pixel divided by its height.

time_base*: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction)*

a packet's `pts` of
`n` means `n * time_base` seconds.

Type:

The unit of the packets' timestamps, in seconds

video_delay*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The number of frames the decoder holds back before it outputs one,
because of B-frames.

width*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The width of the frames, in pixels.