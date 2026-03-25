# Frame

*class*torchcodec.Frame(*data: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *pts_seconds: [float](https://docs.python.org/3/library/functions.html#float)*, *duration_seconds: [float](https://docs.python.org/3/library/functions.html#float)*)[[source]](../_modules/torchcodec/_frame.html#Frame)

A single video frame with associated metadata.

Examples using `Frame`:

![](../_images/sphx_glr_basic_example_thumb.png)

[Decoding a video with VideoDecoder](../generated_examples/decoding/basic_example.html)

Decoding a video with VideoDecoder

data*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

The frame data as (3-D `torch.Tensor`).

duration_seconds*: [float](https://docs.python.org/3/library/functions.html#float)*

The duration of the frame, in seconds (float).

pts_seconds*: [float](https://docs.python.org/3/library/functions.html#float)*

The [pts](../glossary.html#term-pts) of the frame, in seconds (float).