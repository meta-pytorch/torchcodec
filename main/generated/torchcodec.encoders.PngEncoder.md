# PngEncoder

*class*torchcodec.encoders.PngEncoder(*img: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*)[[source]](../_modules/torchcodec/encoders/_image_encoders.html#PngEncoder)

Encoder for PNG images.

Example

```
from torchcodec.encoders import PngEncoder

PngEncoder(img).to_file("image.png")
# or encode to a file-like object or to a tensor, see methods below.
```

Parameters:

**img** (`torch.Tensor`) - The image to encode, a 3-dimensional uint8 tensor
in CHW layout with 1 (grayscale) or 3 (RGB) channels.

to_file(*dest: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, ***, *compression_level: [int](https://docs.python.org/3/library/functions.html#int) = 6*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torchcodec/encoders/_image_encoders.html#PngEncoder.to_file)

Encode the image into a PNG file.

Parameters:

- **dest** (str or `pathlib.Path`) - The path to the output file, e.g.
`image.png`.
- **compression_level** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - zlib compression level between 0
(no compression, fastest) and 9 (max compression, slowest).
Default: 6.

to_file_like(*dest: [RawIOBase](https://docs.python.org/3/library/io.html#io.RawIOBase) | [BufferedIOBase](https://docs.python.org/3/library/io.html#io.BufferedIOBase)*, ***, *compression_level: [int](https://docs.python.org/3/library/functions.html#int) = 6*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torchcodec/encoders/_image_encoders.html#PngEncoder.to_file_like)

Encode the image into a file-like object.

Parameters:

- **dest** - A writable file-like object supporting `write` and `seek`,
such as `io.BytesIO()` or an open file in binary write mode.
- **compression_level** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - zlib compression level between 0
(no compression, fastest) and 9 (max compression, slowest).
Default: 6.

to_tensor(***, *compression_level: [int](https://docs.python.org/3/library/functions.html#int) = 6*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../_modules/torchcodec/encoders/_image_encoders.html#PngEncoder.to_tensor)

Encode the image into raw bytes, as a 1D uint8 tensor.

Parameters:

**compression_level** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - zlib compression level between 0
(no compression, fastest) and 9 (max compression, slowest).
Default: 6.

Returns:

The encoded bytes, a 1D uint8 tensor.

Return type:

[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)