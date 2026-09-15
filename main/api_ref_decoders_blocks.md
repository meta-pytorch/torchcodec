# torchcodec.decoders._blocks

Warning

**The Blocks APIs are under active construction.** They are private and
unreleased. Signatures and semantics may change without notice.

For a tutorial, see:
[Blocks: build your own decoding pipeline](generated_examples/decoding/blocks.html#sphx-glr-generated-examples-decoding-blocks-py).

## Demuxing

| [`Demuxer`](generated/torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer) | Reads one or more video and audio streams from a container, and produces their compressed [`Packet`](generated/torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet)s. |
| --- | --- |
| [`VideoStream`](generated/torchcodec.decoders._blocks.VideoStream.html#torchcodec.decoders._blocks.VideoStream) | A video stream followed by a [`Demuxer`](generated/torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer). |
| [`AudioStream`](generated/torchcodec.decoders._blocks.AudioStream.html#torchcodec.decoders._blocks.AudioStream) | An audio stream followed by a [`Demuxer`](generated/torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer). |

| [`get_container_metadata`](generated/torchcodec.decoders._blocks.get_container_metadata.html#torchcodec.decoders._blocks.get_container_metadata) | TODO_API_BREAKDOWN DOC |
| --- | --- |

## Decoding

| [`VideoPacketDecoder`](generated/torchcodec.decoders._blocks.VideoPacketDecoder.html#torchcodec.decoders._blocks.VideoPacketDecoder) | Decodes the compressed [`Packet`](generated/torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet)s of one video stream into [`RawFrame`](generated/torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame)s. |
| --- | --- |
| [`AudioPacketDecoder`](generated/torchcodec.decoders._blocks.AudioPacketDecoder.html#torchcodec.decoders._blocks.AudioPacketDecoder) | TODO_API_BREAKDOWN DOC |

## Conversion

| [`ColorConverter`](generated/torchcodec.decoders._blocks.ColorConverter.html#torchcodec.decoders._blocks.ColorConverter) | TODO_API_BREAKDOWN DOC |
| --- | --- |
| [`AudioConverter`](generated/torchcodec.decoders._blocks.AudioConverter.html#torchcodec.decoders._blocks.AudioConverter) | TODO_API_BREAKDOWN DOC |

## Data types

| [`Packet`](generated/torchcodec.decoders._blocks.Packet.html#torchcodec.decoders._blocks.Packet) | One compressed packet of one stream, as a [`Demuxer`](generated/torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer) produced it. |
| --- | --- |
| [`RawFrame`](generated/torchcodec.decoders._blocks.RawFrame.html#torchcodec.decoders._blocks.RawFrame) | One decoded video frame, exactly as the decoder produced it. |

| [`FrameIndex`](generated/torchcodec.decoders._blocks.FrameIndex.html#torchcodec.decoders._blocks.FrameIndex) | What the packets of a video stream say about its frames, as returned by a [scan](glossary.html#term-scan) ([`VideoStream.scan()`](generated/torchcodec.decoders._blocks.VideoStream.html#torchcodec.decoders._blocks.VideoStream.scan)). |
| --- | --- |
| [`RawAudioSamples`](generated/torchcodec.decoders._blocks.RawAudioSamples.html#torchcodec.decoders._blocks.RawAudioSamples) | TODO_API_BREAKDOWN DOC |

## Metadata

| [`DemuxerMetadata`](generated/torchcodec.decoders._blocks.DemuxerMetadata.html#torchcodec.decoders._blocks.DemuxerMetadata) | Container-level metadata, as reported by the header. |
| --- | --- |
| [`ContainerMetadata`](generated/torchcodec.decoders._blocks.ContainerMetadata.html#torchcodec.decoders._blocks.ContainerMetadata) | Metadata of a container and of every stream in it. |
| [`VideoStreamHeaderMetadata`](generated/torchcodec.decoders._blocks.VideoStreamHeaderMetadata.html#torchcodec.decoders._blocks.VideoStreamHeaderMetadata) | Metadata of a single video stream, as reported by the container header. |
| [`AudioStreamHeaderMetadata`](generated/torchcodec.decoders._blocks.AudioStreamHeaderMetadata.html#torchcodec.decoders._blocks.AudioStreamHeaderMetadata) | Metadata of a single audio stream, as reported by the container header. |
| [`StreamMetadata`](generated/torchcodec.decoders._blocks.StreamMetadata.html#torchcodec.decoders._blocks.StreamMetadata) | Metadata of a single stream, as reported by the container header. |