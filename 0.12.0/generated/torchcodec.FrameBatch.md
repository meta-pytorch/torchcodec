# FrameBatch

*class*torchcodec.FrameBatch(*data: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *pts_seconds: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *duration_seconds: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*)[[source]](../_modules/torchcodec/_frame.html#FrameBatch)

Multiple video frames with associated metadata.

The `data` tensor is typically 4D for sequences of frames (NHWC or NCHW),
or 5D for sequences of clips, as returned by the [samplers](../generated_examples/decoding/sampling.html#sphx-glr-generated-examples-decoding-sampling-py). When `data` is 4D (resp. 5D)
the `pts_seconds` and `duration_seconds` tensors are 1D (resp. 2D).

Note

The `pts_seconds` and `duration_seconds` Tensors are always returned
on CPU, even if `data` is on GPU.

Examples using `FrameBatch`:

![](../_images/sphx_glr_basic_example_thumb.png)

[Decoding a video with VideoDecoder](../generated_examples/decoding/basic_example.html)

Decoding a video with VideoDecoder
![](../_images/sphx_glr_parallel_decoding_thumb.jpg)

[Parallel video decoding: multi-processing and multi-threading](../generated_examples/decoding/parallel_decoding.html)

Parallel video decoding: multi-processing and multi-threading
![](../_images/sphx_glr_sampling_thumb.png)

[How to sample video clips](../generated_examples/decoding/sampling.html)

How to sample video clips

data*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

The frames data (`torch.Tensor` of uint8).

duration_seconds*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

The duration of the frame, in seconds (`torch.Tensor` of floats).

pts_seconds*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

The [pts](../glossary.html#term-pts) of the frame, in seconds (`torch.Tensor` of floats).