# AudioStreamMetadata

*class*torchcodec.decoders.AudioStreamMetadata(*duration_seconds_from_header: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *begin_stream_seconds_from_header: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *bit_rate: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *codec: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*, *stream_index: [int](https://docs.python.org/3/library/functions.html#int)*, *duration_seconds: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *begin_stream_seconds: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*, *sample_rate: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*, *num_channels: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*, *sample_format: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*)[[source]](../_modules/torchcodec/_core/_metadata.html#AudioStreamMetadata)

Metadata of a single audio stream.

Examples using `AudioStreamMetadata`:

![](../_images/sphx_glr_audio_decoding_thumb.jpg)

[Decoding audio streams with AudioDecoder](../generated_examples/decoding/audio_decoding.html)

Decoding audio streams with AudioDecoder

begin_stream_seconds*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Beginning of the stream, in seconds (float). Conceptually, this
corresponds to the first frame's [pts](../glossary.html#term-pts). If a [scan](../glossary.html#term-scan) was performed
and `begin_stream_seconds_from_content` is not None, then it is returned.
Otherwise, this value is 0.

begin_stream_seconds_from_header*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Beginning of the stream, in seconds, obtained from the header (float or
None). Usually, this is equal to 0.

bit_rate*: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

Bit rate of the stream, in seconds (float or None).

codec*: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

Codec (str or None).

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

num_channels*: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

The number of channels (1 for mono, 2 for stereo, etc.)

sample_format*: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

The original sample format, as described by FFmpeg. E.g. 'fltp', 's32', etc.

sample_rate*: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

The original sample rate.

stream_index*: [int](https://docs.python.org/3/library/functions.html#int)*

Index of the stream that this metadata refers to (int).