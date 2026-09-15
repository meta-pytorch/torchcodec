# FrameIndex

*class*torchcodec.decoders._blocks.FrameIndex(*is_key_frame: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *_pts: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *_duration: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *_time_base_num: [int](https://docs.python.org/3/builtins/functions.html#int)*, *_time_base_den: [int](https://docs.python.org/3/builtins/functions.html#int)*)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#FrameIndex)

What the packets of a video stream say about its frames, as returned by
a [scan](../glossary.html#term-scan) ([`VideoStream.scan()`](torchcodec.decoders._blocks.VideoStream.html#torchcodec.decoders._blocks.VideoStream.scan)).

Its members come in three shapes:

- **Per-frame tensors**, of shape `[N]` where `N` is the number of
frames: one entry per frame, in presentation order. These are
`is_key_frame`, `pts_seconds` and `duration_seconds`.
- **Keyframe positions**, `key_frame_indices`, of shape `[K]` where
`K` is the number of keyframes. These are indices into the per-frame
tensors, not frames themselves.
- **Stream-level metadata scalars**, one value for the whole stream:
`num_frames_from_content`,
`begin_stream_seconds_from_content`,
`end_stream_seconds_from_content` and
`average_fps_from_content`.

`index_at()` and `key_frame_seconds_for()` are convenience methods
that search those tensors so that you don't have to. Together with
[`Demuxer.seek()`](torchcodec.decoders._blocks.Demuxer.html#torchcodec.decoders._blocks.Demuxer.seek), they are what an exact seek is built from:

```
frame_index = video_stream.scan()

# Reach the frame displayed at 12.5 seconds
target = frame_index.pts_seconds[frame_index.index_at(12.5)]
demuxer.seek(frame_index.key_frame_seconds_for(target))
packet_decoder.reset()
# ... then decode forward, dropping the frames before `target`
```

Everything here is derived from the stream's packets rather than from the
container header, so it is exact where the header is only a claim. That is
what the `_from_content` suffixes mark, against the `_from_header` ones
on [`VideoStream.metadata`](torchcodec.decoders._blocks.VideoStream.html#torchcodec.decoders._blocks.VideoStream.metadata).

Examples using `FrameIndex`:

![](../_images/sphx_glr_blocks_thumb.png)

[Blocks: build your own decoding pipeline](../generated_examples/decoding/blocks.html)

Blocks: build your own decoding pipeline

*property*average_fps_from_content*: [float](https://docs.python.org/3/builtins/functions.html#float)*

The average number of frames per second over the stream.

*property*begin_stream_seconds_from_content*: [float](https://docs.python.org/3/builtins/functions.html#float)*

The [pts](../glossary.html#term-pts) of the first frame.

*property*duration_seconds*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Float64 tensor of shape `[N]`, how long each frame is displayed
for.

*property*end_stream_seconds_from_content*: [float](https://docs.python.org/3/builtins/functions.html#float)*

The time at which the last frame stops being displayed.

This is the largest `pts + duration` across the stream, not the last
frame's own end time. Durations vary, so the frame that finishes last
isn't necessarily the one that starts last.

index_at(*seconds: [float](https://docs.python.org/3/builtins/functions.html#float)*) → [int](https://docs.python.org/3/builtins/functions.html#int)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#FrameIndex.index_at)

The index of the frame being displayed at `seconds`.

A frame is displayed from its own [pts](../glossary.html#term-pts) until that plus its
duration, so this is the frame whose interval contains `seconds`.

Parameters:

**seconds** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) - The timestamp to look up. A value outside the
stream gives the closest frame, i.e. the first or the last one.

Returns:

The index of that frame.

Return type:

[int](https://docs.python.org/3/builtins/functions.html#int)

is_key_frame*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Bool tensor of shape `[N]`, whether each frame is a keyframe.

*property*key_frame_indices*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Int64 tensor of shape `[K]`, the indices of the keyframes.

key_frame_seconds_for(*seconds: [float](https://docs.python.org/3/builtins/functions.html#float)*) → [float](https://docs.python.org/3/builtins/functions.html#float)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#FrameIndex.key_frame_seconds_for)

The timestamp of the last keyframe at or before `seconds`.

Parameters:

**seconds** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) - The timestamp you want to reach.

Returns:

The timestamp to seek to.

Return type:

[float](https://docs.python.org/3/builtins/functions.html#float)

*property*num_frames_from_content*: [int](https://docs.python.org/3/builtins/functions.html#int)*

The number of frames in the stream.

*property*pts_seconds*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Float64 tensor of shape `[N]`, the [pts](../glossary.html#term-pts) of each frame.