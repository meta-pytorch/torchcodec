# decode_png

torchcodec.decoders.decode_png(*source: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [bytes](https://docs.python.org/3/library/stdtypes.html#bytes) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *mode: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['UNCHANGED', 'GRAY', 'GRAY_ALPHA', 'RGB', 'RGB_ALPHA'] | [ImageReadMode](torchcodec.decoders.ImageReadMode.html#torchcodec.decoders.ImageReadMode) = 'RGB'*, *output_dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['auto'] = torch.uint8*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../_modules/torchcodec/decoders/_image_decoders.html#decode_png)

Decode a PNG image into a `CHW` tensor.

Example

```
from torchcodec.decoders import decode_png

img = decode_png("image.png")
```

Parameters:

- **source** (str, `pathlib.Path`, bytes, or `torch.Tensor`) - The encoded PNG data: a path (`str` or `pathlib.Path`), a
`bytes` object, or a 1-D uint8 `torch.Tensor` of the raw encoded
bytes.
- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*or*[*ImageReadMode*](torchcodec.decoders.ImageReadMode.html#torchcodec.decoders.ImageReadMode)*,**optional*) - Desired color mode of the output
image. Can be one of `"UNCHANGED"`, `"GRAY"`, `"GRAY_ALPHA"`,
`"RGB"`, or `"RGB_ALPHA"`. Default is `"RGB"`.
- **output_dtype** (torch.dtype or `"auto"`, optional) - desired dtype of the
output image tensor. Accepted values are `torch.uint8` (default),
`torch.uint16`, and `"auto"`. PNG images can natively store
16-bit samples: `torch.uint16` preserves that precision (8-bit
sources are scaled up, 0-255 -> 0-65535), while `torch.uint8`
scales 16-bit sources down. `"auto"` keeps the source's native bit
depth, yielding uint8 for 8-bit PNGs and uint16 for 16-bit ones.

Returns:

The decoded image, of shape `(C, H, W)`.

Return type:

[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Examples using `decode_png`:

![](../_images/sphx_glr_image_decoding_thumb.jpg)

[Decoding images](../generated_examples/decoding/image_decoding.html)

Decoding images