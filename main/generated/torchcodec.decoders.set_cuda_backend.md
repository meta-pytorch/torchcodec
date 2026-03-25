# set_cuda_backend

torchcodec.decoders.set_cuda_backend(*backend: [str](https://docs.python.org/3/library/stdtypes.html#str)*) → [Generator](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)[[None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None)][[source]](../_modules/torchcodec/decoders/_decoder_utils.html#set_cuda_backend)

Context Manager to set the CUDA backend for [`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder).

This context manager allows you to specify which CUDA backend implementation
to use when creating [`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) instances
with CUDA devices.

Note

**We recommend trying the "beta" backend instead of the default "ffmpeg"
backend!** The beta backend is faster, and will eventually become the
default in future versions. It may have rough edges that we'll polish
over time, but it's already quite stable and ready for adoption. Let us
know what you think!

Only the creation of the decoder needs to be inside the context manager, the
decoding methods can be called outside of it. You still need to pass
`device="cuda"` when creating the
[`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) instance. If a CUDA device isn't
specified, this context manager will have no effect. See example below.

This is thread-safe and async-safe.

Parameters:

**backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) - The CUDA backend to use. Can be "ffmpeg" (default) or
"beta". We recommend trying "beta" as it's faster!

Example

```
>>> with set_cuda_backend("beta"):
... decoder = VideoDecoder("video.mp4", device="cuda")
...
... # Only the decoder creation needs to be part of the context manager.
... # Decoder will now the beta CUDA implementation:
... decoder.get_frame_at(0)
```

Examples using `set_cuda_backend`:

![](../_images/sphx_glr_basic_cuda_example_thumb.png)

[Accelerated video decoding on GPUs with CUDA and NVDEC](../generated_examples/decoding/basic_cuda_example.html)

Accelerated video decoding on GPUs with CUDA and NVDEC