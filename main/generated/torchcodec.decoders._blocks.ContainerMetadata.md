# ContainerMetadata

*class*torchcodec.decoders._blocks.ContainerMetadata(*duration_seconds_from_header: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *bit_rate_from_header: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *best_video_stream_index: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *best_audio_stream_index: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *streams: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[StreamMetadata](torchcodec.decoders._blocks.StreamMetadata.html#torchcodec.decoders._blocks.StreamMetadata)]*)[[source]](../_modules/torchcodec/_core/_metadata.html#ContainerMetadata)

Metadata of a container and of every stream in it.

Unlike [`DemuxerMetadata`](torchcodec.decoders._blocks.DemuxerMetadata.html#torchcodec.decoders._blocks.DemuxerMetadata), which describes the container alone, this
also lists the streams, including the ones that cannot be decoded.

best_audio_stream_index*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Index of the [best stream](../glossary.html#term-best-stream) of audio type (int or None).

best_video_stream_index*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Index of the [best stream](../glossary.html#term-best-stream) of video type (int or None).

bit_rate_from_header*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Overall bit rate of the container (float or None).

duration_seconds_from_header*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Duration of the container, in seconds, obtained from the header (float
or None).

streams*: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[StreamMetadata](torchcodec.decoders._blocks.StreamMetadata.html#torchcodec.decoders._blocks.StreamMetadata)]*

One entry per stream in the file, indexed by stream index. Streams that
are neither video nor audio (subtitles, data) are plain
[`StreamMetadata`](torchcodec.decoders._blocks.StreamMetadata.html#torchcodec.decoders._blocks.StreamMetadata): you can see them, you cannot decode them.