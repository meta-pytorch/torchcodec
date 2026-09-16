# get_container_metadata

torchcodec.decoders._blocks.get_container_metadata(*source: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [RawIOBase](https://docs.python.org/3/library/io.html#io.RawIOBase) | BufferedReader*) → [ContainerMetadata](torchcodec.decoders._blocks.ContainerMetadata.html#torchcodec.decoders._blocks.ContainerMetadata)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#get_container_metadata)

Describe a container and every stream in it, without decoding anything.

```
metadata = get_container_metadata("video.mp4")
metadata.duration_seconds_from_header
metadata.streams[metadata.best_video_stream_index].width
```

Only the header is read, so this is what to reach for before you know what a
file holds, and therefore which streams to ask a [`Demuxer`](torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer) for.

Parameters:

**source** (str, `Pathlib.path`, bytes, `torch.Tensor` or file-like object) - The source of the media, as for [`Demuxer`](torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer).

Returns:

The container's own metadata, plus one entry per
stream, indexed by stream index.

Return type:

[ContainerMetadata](torchcodec.decoders._blocks.ContainerMetadata.html#torchcodec.decoders._blocks.ContainerMetadata)

Examples using `get_container_metadata`:

![](../_images/sphx_glr_basics_thumb.png)

[Blocks: build your own decoding pipeline](../generated_examples/blocks/basics.html)

Blocks: build your own decoding pipeline