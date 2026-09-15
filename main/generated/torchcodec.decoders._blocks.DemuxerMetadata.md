# DemuxerMetadata

*class*torchcodec.decoders._blocks.DemuxerMetadata(*duration_seconds_from_header: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *bit_rate_from_header: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *best_video_stream_index: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*, *best_audio_stream_index: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*)[[source]](../_modules/torchcodec/_core/_metadata.html#DemuxerMetadata)

Container-level metadata, as reported by the header.

This is what a [`Demuxer`](torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer) can say about
the container itself.

best_audio_stream_index*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Index of the [best stream](../glossary.html#term-best-stream) of audio type (int or None).

best_video_stream_index*: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Index of the [best stream](../glossary.html#term-best-stream) of video type (int or None).

bit_rate_from_header*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Overall bit rate of the container (float or None).

duration_seconds_from_header*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Duration of the container, in seconds, obtained from the header (float
or None).