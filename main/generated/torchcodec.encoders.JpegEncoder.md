# JpegEncoder

*class*torchcodec.encoders.JpegEncoder(*img: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*)[[source]](../_modules/torchcodec/encoders/_image_encoders.html#JpegEncoder)

Encoder for JPEG images.

Example

```
from torchcodec.encoders import JpegEncoder

JpegEncoder(img).to_file("image.jpg")
# or encode to a file-like object or to a tensor, see methods below.
```

Parameters:

**img** (`torch.Tensor`) - The image to encode, a 3-dimensional uint8 tensor
in CHW layout with 1 (grayscale) or 3 (RGB) channels. If on a CUDA
device, encoding is performed on the GPU with nvJPEG, and only
3-channel RGB is supported.

to_file(*dest: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, ***, *quality: [int](https://docs.python.org/3/library/functions.html#int) = 75*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torchcodec/encoders/_image_encoders.html#JpegEncoder.to_file)

Encode the image into a JPEG file.

Parameters:

- **dest** (str or `pathlib.Path`) - The path to the output file, e.g.
`image.jpg`.
- **quality** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - Quality of the output, between 1 and 100.
Higher means better quality and larger file size. Default: 75.

to_file_like(*dest: [RawIOBase](https://docs.python.org/3/library/io.html#io.RawIOBase) | [BufferedIOBase](https://docs.python.org/3/library/io.html#io.BufferedIOBase)*, ***, *quality: [int](https://docs.python.org/3/library/functions.html#int) = 75*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torchcodec/encoders/_image_encoders.html#JpegEncoder.to_file_like)

Encode the image into a file-like object.

Parameters:

- **dest** - A writable file-like object supporting `write` and `seek`,
such as `io.BytesIO()` or an open file in binary write mode.
- **quality** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - Quality of the output, between 1 and 100.
Higher means better quality and larger file size. Default: 75.

to_tensor(***, *quality: [int](https://docs.python.org/3/library/functions.html#int) = 75*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../_modules/torchcodec/encoders/_image_encoders.html#JpegEncoder.to_tensor)

Encode the image into raw bytes, as a 1D uint8 tensor.

The returned tensor is on the same device as the input image (a CUDA
input yields a CUDA tensor).

Parameters:

**quality** ([*int*](https://docs.python.org/3/library/functions.html#int)*,**optional*) - Quality of the output, between 1 and 100.
Higher means better quality and larger file size. Default: 75.

Returns:

The encoded bytes, a 1D uint8 tensor on the same
device as the input image.

Return type:

[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)