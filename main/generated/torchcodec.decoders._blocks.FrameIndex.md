# FrameIndex

*class*torchcodec.decoders._blocks.FrameIndex(*is_key_frame: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *_pts: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *_duration: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *_time_base_num: [int](https://docs.python.org/3/builtins/functions.html#int)*, *_time_base_den: [int](https://docs.python.org/3/builtins/functions.html#int)*)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#FrameIndex)

TODO_API_BREAKDOWN DOC

Examples using `FrameIndex`:

![](../_images/sphx_glr_blocks_thumb.png)

[Blocks: build your own decoding pipeline](../generated_examples/decoding/blocks.html)

Blocks: build your own decoding pipeline

*property*average_fps_from_content*: [float](https://docs.python.org/3/builtins/functions.html#float)*

*property*begin_stream_seconds_from_content*: [float](https://docs.python.org/3/builtins/functions.html#float)*

*property*duration_seconds*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

*property*end_stream_seconds_from_content*: [float](https://docs.python.org/3/builtins/functions.html#float)*

index_at(*seconds: [float](https://docs.python.org/3/builtins/functions.html#float)*) → [int](https://docs.python.org/3/builtins/functions.html#int)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#FrameIndex.index_at)

is_key_frame*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

*property*key_frame_indices*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

key_frame_seconds_for(*seconds: [float](https://docs.python.org/3/builtins/functions.html#float)*) → [float](https://docs.python.org/3/builtins/functions.html#float)[[source]](../_modules/torchcodec/decoders/_blocks/_demuxer.html#FrameIndex.key_frame_seconds_for)

*property*num_frames_from_content*: [int](https://docs.python.org/3/builtins/functions.html#int)*

*property*pts_seconds*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*