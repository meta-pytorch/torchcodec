# AudioStreamHeaderMetadata

*class*torchcodec.decoders._blocks.AudioStreamHeaderMetadata(*duration_seconds_from_header: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *begin_stream_seconds_from_header: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *bit_rate: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *codec: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *stream_index: [int](https://docs.python.org/3/builtins/functions.html#int)*, *media_type: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*, *sample_rate: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *num_channels: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *sample_format: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*)[[source]](../_modules/torchcodec/_core/_metadata.html#AudioStreamHeaderMetadata)

Metadata of a single audio stream, as reported by the container header.

begin_stream_seconds_from_header*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Beginning of the stream, in seconds, obtained from the header (float or
None). Usually, this is equal to 0.

bit_rate*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Bit rate of the stream, in seconds (float or None).

codec*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Codec (str or None).

duration_seconds_from_header*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Duration of the stream, in seconds, obtained from the header (float or
None). This could be inaccurate.

media_type*: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Type of media the stream carries (str). One of `"video"`, `"audio"`,
`"subtitle"`, `"data"`, `"attachment"` or `"unknown"`.

num_channels*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The number of channels (1 for mono, 2 for stereo, etc.)

sample_format*: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The original sample format, as described by FFmpeg. E.g. 'fltp', 's32', etc.

sample_rate*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The original sample rate.

stream_index*: [int](https://docs.python.org/3/builtins/functions.html#int)*

Index of the stream that this metadata refers to (int).