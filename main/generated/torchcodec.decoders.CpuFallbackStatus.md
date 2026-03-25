# CpuFallbackStatus

*class*torchcodec.decoders.CpuFallbackStatus(*status_known: [bool](https://docs.python.org/3/library/functions.html#bool) = False*)[[source]](../_modules/torchcodec/decoders/_video_decoder.html#CpuFallbackStatus)

Information about CPU fallback status.

This class tracks whether the decoder fell back to CPU decoding.
Users should not instantiate this class directly; instead, access it
via the `VideoDecoder.cpu_fallback` attribute.

Usage:

- Use `str(cpu_fallback_status)` or `print(cpu_fallback_status)` to see the cpu fallback status
- Use `if cpu_fallback_status:` to check if any fallback occurred

Examples using `CpuFallbackStatus`:

![](../_images/sphx_glr_basic_cuda_example_thumb.png)

[Accelerated video decoding on GPUs with CUDA and NVDEC](../generated_examples/decoding/basic_cuda_example.html)

Accelerated video decoding on GPUs with CUDA and NVDEC

status_known*: [bool](https://docs.python.org/3/library/functions.html#bool)**= False*

Whether the fallback status has been determined.
For the Beta CUDA backend (see [`set_cuda_backend()`](torchcodec.decoders.set_cuda_backend.html#torchcodec.decoders.set_cuda_backend)),
this is always `True` immediately after decoder creation.
For the FFmpeg CUDA backend, this becomes `True` after decoding
the first frame.