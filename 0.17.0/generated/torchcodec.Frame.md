# Frame

*class*torchcodec.Frame(*data: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *pts_seconds: [float](https://docs.python.org/3/builtins/functions.html#float)*, *duration_seconds: [float](https://docs.python.org/3/builtins/functions.html#float)*)[[source]](../_modules/torchcodec/_frame.html#Frame)

A single video frame with associated metadata.

Examples using `Frame`:

![](../_images/sphx_glr_basic_example_thumb.png)

[Decoding a video with VideoDecoder](../generated_examples/decoding/basic_example.html)

Decoding a video with VideoDecoder
![](../_images/sphx_glr_basics_thumb.jpg)

[Build your own decoding pipeline](../generated_examples/low_level/basics.html)

Build your own decoding pipeline
![](../_images/sphx_glr_pipelines_thumb.jpg)

[Multi-threaded decoding pipelines](../generated_examples/low_level/pipelines.html)

Multi-threaded decoding pipelines

data*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

The frame data as (3-D `torch.Tensor`).

duration_seconds*: [float](https://docs.python.org/3/builtins/functions.html#float)*

The duration of the frame, in seconds (float).

pts_seconds*: [float](https://docs.python.org/3/builtins/functions.html#float)*

The [pts](../glossary.html#term-pts) of the frame, in seconds (float).