# decode_image

torchcodec.decoders.decode_image(*source: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [bytes](https://docs.python.org/3/library/stdtypes.html#bytes) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *mode: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['UNCHANGED', 'GRAY', 'GRAY_ALPHA', 'RGB', 'RGB_ALPHA'] | [ImageReadMode](torchcodec.decoders.ImageReadMode.html#torchcodec.decoders.ImageReadMode) = 'RGB'*, *output_dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['auto'] = torch.uint8*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../_modules/torchcodec/decoders/_image_decoders.html#decode_image)

Decode an image into a `[N]CHW` tensor, detecting the format automatically.

The format is detected from the encoded data (not the file extension), and
decoding is delegated to the matching format-specific decoder. Supported
formats are JPEG, PNG, WebP, GIF, AVIF and HEIC (requires `libheif`). The
output shape is `(C, H, W)` for a single image and `(N, C, H, W)` for
animated or multi-image formats (WebP, GIF, AVIF, HEIC).

For finer control, or for format-specific options (e.g. `device` for CUDA
JPEG decoding, `num_threads` for AVIF), use the dedicated decoders
directly: [`decode_jpeg()`](torchcodec.decoders.decode_jpeg.html#torchcodec.decoders.decode_jpeg), [`decode_png()`](torchcodec.decoders.decode_png.html#torchcodec.decoders.decode_png), [`decode_webp()`](torchcodec.decoders.decode_webp.html#torchcodec.decoders.decode_webp),
[`decode_gif()`](torchcodec.decoders.decode_gif.html#torchcodec.decoders.decode_gif), [`decode_avif()`](torchcodec.decoders.decode_avif.html#torchcodec.decoders.decode_avif), [`decode_heic()`](torchcodec.decoders.decode_heic.html#torchcodec.decoders.decode_heic).

Example

```
from torchcodec.decoders import decode_image

jpeg_img = decode_image("image.jpg")
png_img = decode_image("image.png")
```

Parameters:

- **source** (str, `pathlib.Path`, bytes, or `torch.Tensor`) - The encoded image data: a path (`str` or `pathlib.Path`), a
`bytes` object, or a 1-D uint8 `torch.Tensor` of the raw encoded
bytes.
- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*or*[*ImageReadMode*](torchcodec.decoders.ImageReadMode.html#torchcodec.decoders.ImageReadMode)*,**optional*) - Desired color mode of the output
image. Can be one of `"UNCHANGED"`, `"GRAY"`, `"GRAY_ALPHA"`,
`"RGB"`, or `"RGB_ALPHA"`. Default is `"RGB"`.
- **output_dtype** (torch.dtype or `"auto"`, optional) - desired dtype of the
output image tensor. Accepted values are `torch.uint8` (default),
`torch.uint16`, and `"auto"`. Formats that can carry more than 8
bits per channel (PNG, AVIF, HEIC) preserve that precision with
`torch.uint16` and `"auto"`. See the format-specific decoders
for details.

Returns:

The decoded image, of shape `[N]CHW`.

Return type:

[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)