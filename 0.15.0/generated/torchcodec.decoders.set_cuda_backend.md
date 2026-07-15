# set_cuda_backend

torchcodec.decoders.set_cuda_backend(*backend: [str](https://docs.python.org/3/library/stdtypes.html#str)*) → [Generator](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)[[None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None), [None](https://docs.python.org/3/library/constants.html#None)][[source]](../_modules/torchcodec/decoders/_decoder_utils.html#set_cuda_backend)

Context Manager to set the CUDA backend for [`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder).

This context manager allows you to specify which CUDA backend implementation
to use when creating [`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) instances
with CUDA devices.

The default is `"nvdec"`. An `"ffmpeg"` backend is also available.

Only the creation of the decoder needs to be inside the context manager, the
decoding methods can be called outside of it. You still need to pass
`device="cuda"` when creating the
[`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) instance. If a CUDA device isn't
specified, this context manager will have no effect. See example below.

This is thread-safe and async-safe.

Parameters:

**backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) - The CUDA backend to use. Can be `"nvdec"` (default) or
`"ffmpeg"`.

Example

```
>>> with set_cuda_backend("ffmpeg"):
... decoder = VideoDecoder("video.mp4", device="cuda")
...
... # Only the decoder creation needs to be part of the context manager.
... # Decoder will use the FFmpeg CUDA implementation:
... decoder.get_frame_at(0)
```