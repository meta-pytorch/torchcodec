# decode_jpeg

torchcodec.decoders.decode_jpeg(*source: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [bytes](https://docs.python.org/3/library/stdtypes.html#bytes) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [list](https://docs.python.org/3/library/stdtypes.html#list)*, ***, *mode: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['UNCHANGED', 'GRAY', 'GRAY_ALPHA', 'RGB', 'RGB_ALPHA'] | [ImageReadMode](torchcodec.decoders.ImageReadMode.html#torchcodec.decoders.ImageReadMode) = 'RGB'*, *output_dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['auto'] = torch.uint8*, *device: [str](https://docs.python.org/3/library/stdtypes.html#str) | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) = 'cpu'*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [list](https://docs.python.org/3/library/stdtypes.html#list)[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../_modules/torchcodec/decoders/_image_decoders.html#decode_jpeg)

Decode a JPEG image into a `CHW` tensor, on CPU or CUDA.

Note

For CUDA decoding, prefer passing a batch (a list of sources) in a
single call: the whole batch is decoded in one nvJPEG call, which is
much faster than decoding images one at a time.
Passing a batch of sources is supported on CPU too, but it won't be
faster than decoding them one at a time.

Example

```
from torchcodec.decoders import decode_jpeg

img = decode_jpeg("image.jpg")
img = decode_jpeg("image.jpg", device="cuda") # decode on GPU
```

Parameters:

- **source** (str, `pathlib.Path`, bytes, `torch.Tensor`, or list of these) - The encoded JPEG data: a path (`str` or `pathlib.Path`), a
`bytes` object, or a 1-D uint8 `torch.Tensor` of the raw encoded
bytes. Pass a list (or tuple) to decode a batch, in which case a list of
tensors is returned instead of a single tensor. The encoded bytes must
live on CPU, even when decoding to a CUDA device.
- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*or*[*ImageReadMode*](torchcodec.decoders.ImageReadMode.html#torchcodec.decoders.ImageReadMode)*,**optional*) - Desired color mode of the output
image. Can be one of `"UNCHANGED"`, `"GRAY"`, `"GRAY_ALPHA"`,
`"RGB"`, or `"RGB_ALPHA"`. Default is `"RGB"`.
- **output_dtype** (torch.dtype or `"auto"`, optional) - desired dtype of the
output image tensor. Accepted values are `torch.uint8` (default),
`torch.uint16`, and `"auto"`. Since JPEG is an 8-bit format,
`"auto"` and `torch.uint8` are equivalent. `torch.uint16`
emulates a 16-bit output by scaling the 8-bit values to the full
16-bit range (0-255 -> 0-65535).
- **device** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device to decode on, `"cpu"`
(default) or a CUDA device. CUDA decoding uses nvJPEG. We recommend
passing a batch of sources when decoding on CUDA, for speed.

Returns:

The decoded
image(s). A single tensor for a single source, or a list of tensors for
a batch.

Return type:

torch.Tensor or list of torch.Tensor of shape `C, H, W`

Examples using `decode_jpeg`:

![](../_images/sphx_glr_image_decoding_thumb.png)

[Decoding images](../generated_examples/decoding/image_decoding.html)

Decoding images