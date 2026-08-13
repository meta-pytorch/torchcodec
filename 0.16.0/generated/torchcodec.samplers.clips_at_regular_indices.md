# clips_at_regular_indices

torchcodec.samplers.clips_at_regular_indices(*decoder: [VideoDecoder](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder)*, ***, *num_clips: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *num_frames_per_clip: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *num_indices_between_frames: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *sampling_range_start: [int](https://docs.python.org/3/library/functions.html#int) = 0*, *sampling_range_end: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *policy: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['repeat_last', 'wrap', 'error'] = 'repeat_last'*) → [FrameBatch](torchcodec.FrameBatch.html#torchcodec.FrameBatch)[[source]](../_modules/torchcodec/samplers/_index_based.html#clips_at_regular_indices)

Sample [clips](../glossary.html#term-clips) at regular (equally-spaced) indices.

Parameters:

- **decoder** ([*VideoDecoder*](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder)) - The [`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder)
instance to sample clips from.
- **num_clips** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The number of clips to return. Default: 1.
- **num_frames_per_clip** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The number of frames per clips. Default: 1.
- **num_indices_between_frames** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The number of indices between
the frames *within* a clip. Default: 1, which means frames are
consecutive. This is sometimes refered-to as "dilation".
- **sampling_range_start** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The start of the sampling range,
which defines the first index that a clip may *start* at. Default:
0, i.e. the start of the video.
- **sampling_range_end** ([*int*](https://docs.python.org/3/library/functions.html#int)*or**None**,**optional*) - The end of the sampling
range, which defines the last index that a clip may *start* at. This
value is exclusive, i.e. a clip may only start within
[`sampling_range_start`, `sampling_range_end`). If None
(default), the value is set automatically such that the clips never
span beyond the end of the video. For example if the last valid
index in a video is 99 and the clips span 10 frames, this value is
set to 99 - 10 + 1 = 90. Negative values are accepted and are
equivalent to `len(video) - val`. When a clip spans beyond the end
of the video, the `policy` parameter defines how to construct such
clip.
- **policy** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**optional*) - 

Defines how to construct clips that span beyond
the end of the video. This is best described with an example:
assuming the last valid index in a video is 99, and a clip was
sampled to start at index 95, with `num_frames_per_clip=5` and
`num_indices_between_frames=2`, the indices of the frames in the
clip are supposed to be [95, 97, 99, 101, 103]. But 101 and 103 are
invalid indices, so the `policy` parameter defines how to replace
those frames, with valid indices:

- "repeat_last": repeats the last valid frame of the clip. We would
get [95, 97, 99, 99, 99].
- "wrap": wraps around to the beginning of the clip. We would get
[95, 97, 99, 95, 97].
- "error": raises an error.

Default is "repeat_last". Note that when `sampling_range_end=None`
(default), this policy parameter is unlikely to be relevant.

Returns:

The sampled [clips](../glossary.html#term-clips), as a 5D [`FrameBatch`](torchcodec.FrameBatch.html#torchcodec.FrameBatch).
The shape of the `data` field is (`num_clips`,
`num_frames_per_clips`, ...) where ... is (H, W, C) or (C, H, W)
depending on the `dimension_order` parameter of
[`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder). The shape of the
`pts_seconds` and `duration_seconds` fields is (`num_clips`,
`num_frames_per_clips`).

Return type:

[FrameBatch](torchcodec.FrameBatch.html#torchcodec.FrameBatch)

Examples using `clips_at_regular_indices`:

![](../_images/sphx_glr_sampling_thumb.png)

[How to sample video clips](../generated_examples/decoding/sampling.html)

How to sample video clips
![](../_images/sphx_glr_transforms_thumb.png)

[Decoder Transforms: Applying transforms during decoding](../generated_examples/decoding/transforms.html)

Decoder Transforms: Applying transforms during decoding