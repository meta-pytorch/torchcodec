# clips_at_random_timestamps

torchcodec.samplers.clips_at_random_timestamps(*decoder*, ***, *num_clips: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *num_frames_per_clip: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *seconds_between_frames: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *sampling_range_start: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *sampling_range_end: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *policy: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['repeat_last', 'wrap', 'error'] = 'repeat_last'*) → [FrameBatch](torchcodec.FrameBatch.html#torchcodec.FrameBatch)[[source]](../_modules/torchcodec/samplers/_time_based.html#clips_at_random_timestamps)

Sample [clips](../glossary.html#term-clips) at random timestamps.

Parameters:

- **decoder** ([*VideoDecoder*](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder)) - The [`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder)
instance to sample clips from.
- **num_clips** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The number of clips to return. Default: 1.
- **num_frames_per_clip** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - The number of frames per clips. Default: 1.
- **seconds_between_frames** ([*float*](https://docs.python.org/3/library/functions.html#float)*or**None**,**optional*) - The time (in seconds)
between each frame within a clip. More accurately, this defines the
time between the *frame sampling point*, i.e. the timestamps at
which we sample the frames. Because frames span intervals in time ,
the resulting start of frames within a clip may not be exactly
spaced by `seconds_between_frames` - but on average, they will be.
Default is None, which is set to the average frame duration
(`1/average_fps`).
- **sampling_range_start** ([*float*](https://docs.python.org/3/library/functions.html#float)*or**None**,**optional*) - The start of the
sampling range, which defines the first timestamp (in seconds) that
a clip may *start* at. Default: None, which corresponds to the start
of the video. (Note: some videos start at negative values, which is
why the default is not 0).
- **sampling_range_end** ([*float*](https://docs.python.org/3/library/functions.html#float)*or**None**,**optional*) - The end of the sampling
range, which defines the last timestamp (in seconds) that a clip may
*start* at. This value is exclusive, i.e. a clip may only start within
[`sampling_range_start`, `sampling_range_end`). If None
(default), the value is set automatically such that the clips never
span beyond the end of the video, i.e. it is set to
`end_video_seconds - (num_frames_per_clip - 1) *
seconds_between_frames`. When a clip spans beyond the end of the
video, the `policy` parameter defines how to construct such clip.
- **policy** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,**optional*) - 

Defines how to construct clips that span beyond
the end of the video. This is best described with an example:
assuming the last valid (seekable) timestamp in a video is 10.9, and
a clip was sampled to start at timestamp 10.5, with
`num_frames_per_clip=5` and `seconds_between_frames=0.2`, the
sampling timestamps of the frames in the clip are supposed to be
[10.5, 10.7, 10.9, 11.1, 11.2]. But 11.1 and 11.2 are invalid
timestamps, so the `policy` parameter defines how to replace those
frames, with valid sampling timestamps:

- "repeat_last": repeats the last valid frame of the clip. We would
get frames sampled at timestamps [10.5, 10.7, 10.9, 10.9, 10.9].
- "wrap": wraps around to the beginning of the clip. We would get
frames sampled at timestamps [10.5, 10.7, 10.9, 10.5, 10.7].
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

Examples using `clips_at_random_timestamps`:

![](../_images/sphx_glr_sampling_thumb.png)

[How to sample video clips](../generated_examples/decoding/sampling.html)

How to sample video clips