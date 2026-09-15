# StreamMetadata

*class*torchcodec.decoders._blocks.StreamMetadata(*duration_seconds_from_header: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *begin_stream_seconds_from_header: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *bit_rate: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *codec: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *stream_index: [int](https://docs.python.org/3/builtins/functions.html#int)*)[[source]](../_modules/torchcodec/_core/_metadata.html#StreamMetadata)

Metadata of a single stream, as reported by the container header.

This is everything that is known about a stream without looking at its
content. Streams that are neither video nor audio (subtitles, data) are
described by this class and nothing more.

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

stream_index*: [int](https://docs.python.org/3/builtins/functions.html#int)*

Index of the stream that this metadata refers to (int).