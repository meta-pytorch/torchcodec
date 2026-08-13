# decode_gif

torchcodec.decoders.decode_gif(*source: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [bytes](https://docs.python.org/3/library/stdtypes.html#bytes) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *mode: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['UNCHANGED', 'GRAY', 'GRAY_ALPHA', 'RGB', 'RGB_ALPHA'] | [ImageReadMode](torchcodec.decoders.ImageReadMode.html#torchcodec.decoders.ImageReadMode) = 'RGB'*, *output_dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['auto'] = torch.uint8*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../_modules/torchcodec/decoders/_image_decoders.html#decode_gif)

Decode a GIF image into a `[N]CHW` tensor.

The output shape is `(C, H, W)` for a still GIF and `(N, C, H, W)` for
an animated one (N frames).

Example

```
from torchcodec.decoders import decode_gif

img = decode_gif("image.gif")
```

Parameters:

- **source** (str, `pathlib.Path`, bytes, or `torch.Tensor`) - The encoded GIF data: a path (`str` or `pathlib.Path`), a
`bytes` object, or a 1-D uint8 `torch.Tensor` of the raw encoded
bytes.
- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*or*[*ImageReadMode*](torchcodec.decoders.ImageReadMode.html#torchcodec.decoders.ImageReadMode)*,**optional*) - Desired color mode of the output
image. Can be one of `"UNCHANGED"`, `"GRAY"`, `"GRAY_ALPHA"`,
`"RGB"`, or `"RGB_ALPHA"`. Default is `"RGB"`.
- **output_dtype** (torch.dtype or `"auto"`, optional) - desired dtype of the
output image tensor. Accepted values are `torch.uint8` (default),
`torch.uint16`, and `"auto"`. Since GIF is an 8-bit format,
`"auto"` and `torch.uint8` are equivalent. `torch.uint16`
emulates a 16-bit output by scaling the 8-bit values to the full
16-bit range (0-255 -> 0-65535).

Returns:

The decoded image, of shape `(C, H, W)` (still) or
`(N, C, H, W)` (animated).

Return type:

[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Examples using `decode_gif`:

![](../_images/sphx_glr_image_decoding_thumb.png)

[Decoding images](../generated_examples/decoding/image_decoding.html)

Decoding images