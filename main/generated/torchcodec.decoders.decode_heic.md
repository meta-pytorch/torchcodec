# decode_heic

torchcodec.decoders.decode_heic(*source: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [bytes](https://docs.python.org/3/library/stdtypes.html#bytes) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *mode: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['UNCHANGED', 'GRAY', 'GRAY_ALPHA', 'RGB', 'RGB_ALPHA'] | [ImageReadMode](torchcodec.decoders.ImageReadMode.html#torchcodec.decoders.ImageReadMode) = 'RGB'*, *output_dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['auto'] = torch.uint8*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../_modules/torchcodec/decoders/_image_decoders.html#decode_heic)

Decode an HEIC/HEIF image into a `[N]CHW` tensor - requires `libheif`!

The output shape is `(C, H, W)` for a single-image HEIC and
`(N, C, H, W)` for a multi-image one. All images must share the same
dimensions and bit depth.

Important

HEIC decoding requires **libheif** to be installed and discoverable at
runtime. TorchCodec does not bundle it (libheif is LGPL): install it via
e.g. `conda install -c conda-forge libheif`.

Example

```
from torchcodec.decoders import decode_heic

img = decode_heic("image.heic")
```

Parameters:

- **source** (str, `pathlib.Path`, bytes, or `torch.Tensor`) - The encoded HEIC/HEIF data: a path (`str` or `pathlib.Path`), a
`bytes` object, or a 1-D uint8 `torch.Tensor` of the raw encoded
bytes.
- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*or*[*ImageReadMode*](torchcodec.decoders.ImageReadMode.html#torchcodec.decoders.ImageReadMode)*,**optional*) - Desired color mode of the output
image. Can be one of `"UNCHANGED"`, `"GRAY"`, `"GRAY_ALPHA"`,
`"RGB"`, or `"RGB_ALPHA"`. Default is `"RGB"`.
- **output_dtype** (torch.dtype or `"auto"`, optional) - desired dtype of the
output image tensor. Accepted values are `torch.uint8` (default),
`torch.uint16`, and `"auto"`. HEIC can store more than 8 bits per
channel (e.g. 10- or 12-bit sources). `torch.uint16` always scales
the samples up to fill the full 16-bit range `[0, 65535]` (8-bit
0-255, 10-bit 0-1023 and 12-bit 0-4095 sources are all upscaled),
while `torch.uint8` scales higher-bit sources down. `"auto"`
yields uint8 for 8-bit HEICs and uint16 (again filling `[0, 65535]`)
for higher-bit ones.

Returns:

The decoded image, of shape `(C, H, W)` (single-image)
or `(N, C, H, W)` (multi-image).

Return type:

[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Examples using `decode_heic`:

![](../_images/sphx_glr_image_decoding_thumb.jpg)

[Decoding images](../generated_examples/decoding/image_decoding.html)

Decoding images