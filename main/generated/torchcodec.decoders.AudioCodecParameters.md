# AudioCodecParameters

*class*torchcodec.decoders.AudioCodecParameters(*codec: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*, ***, *time_base: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction)*, *extradata: [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *codec_tag: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *profile: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *level: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *bit_rate: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *bits_per_coded_sample: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *bits_per_raw_sample: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *coded_side_data: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *sample_format: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *sample_rate: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *channel_layout: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *block_align: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *frame_size: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *initial_padding: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *trailing_padding: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *seek_preroll: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)[[source]](../_modules/torchcodec/decoders/_blocks/_codec_parameters.html#AudioCodecParameters)

What an [`AudioPacketDecoder`](torchcodec.decoders.AudioPacketDecoder.html#torchcodec.decoders.AudioPacketDecoder) needs to know to decode the packets
of an audio stream.

If you have a [`Demuxer`](torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer), [`AudioStream.codec_parameters`](torchcodec.decoders.AudioStream.html#torchcodec.decoders.AudioStream.codec_parameters) gives
you the parameters of a stream. They can be pickled, e.g. to build the
decoder in another process. To decode packets that don't come from a
`Demuxer`, build them yourself:

```
codec_parameters = AudioCodecParameters(
 "aac",
 time_base=Fraction(1, 48_000),
 sample_rate=48_000,
 channel_layout="stereo",
 extradata=extradata,
)
decoder = AudioPacketDecoder(codec_parameters)
```

The fields mirror those of FFmpeg's [AVCodecParameters](https://ffmpeg.org/doxygen/trunk/structAVCodecParameters.html), with the
same names and meanings. FFmpeg's names stand for its enums, e.g.
`sample_format="fltp"`.

Which fields you need depends on the codec: printing the parameters a
`Demuxer` reports for a similar file shows what FFmpeg uses.

Note

The parameters are meant to be passed to a decoder or stored, not read
as metadata: use [`AudioStream.metadata`](torchcodec.decoders.AudioStream.html#torchcodec.decoders.AudioStream.metadata) for that. The values a
`Demuxer` reports are FFmpeg's, and can change across FFmpeg and
TorchCodec versions. The fields' names and meanings are stable.

Examples using `AudioCodecParameters`:

![](../_images/sphx_glr_basics_thumb.jpg)

[Build your own decoding pipeline](../generated_examples/low_level/basics.html)

Build your own decoding pipeline

bit_rate*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The average bit rate, in bits per second.

bits_per_coded_sample*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The number of bits per sample in the compressed data, for the codecs
that need it (mostly uncompressed and PCM codecs).

bits_per_raw_sample*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The number of bits per sample in the decoded data, when it's not implied
by the pixel or sample format, e.g. 10 for 10-bit video.

block_align*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The number of bytes per coded audio frame, for the codecs that need it
(e.g. PCM and ADPCM codecs).

channel_layout*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

FFmpeg's description of the channel layout, e.g. `"stereo"` or
`"5.1"`. `ffmpeg -layouts` lists them.

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

extradata*: [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The codec's setup data that isn't in the packets, e.g. the `avcC` box
of H.264 in MP4. Not all codecs and streams need it.

frame_size*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The number of samples per channel in a frame, for the codecs whose
frames all have the same size.

initial_padding*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The number of samples the encoder added at the start of the audio,
which aren't part of the original audio.

level*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The codec's level.

profile*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The codec's profile, as one of FFmpeg's `AV_PROFILE_*` values.

sample_format*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

FFmpeg's name of the decoded samples' format, e.g. `"fltp"`.
`ffmpeg -sample_fmts` lists them.

sample_rate*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The sample rate, in Hz.

seek_preroll*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The number of samples to skip after a discontinuity, e.g. a seek.

time_base*: [Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction)*

a packet's `pts` of
`n` means `n * time_base` seconds.

Type:

The unit of the packets' timestamps, in seconds

trailing_padding*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)**= None*

The number of samples the encoder added at the end of the audio, which
aren't part of the original audio.