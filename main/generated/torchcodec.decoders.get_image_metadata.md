# get_image_metadata

torchcodec.decoders.get_image_metadata(*source: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [ImageMetadata](torchcodec.decoders.ImageMetadata.html#torchcodec.decoders.ImageMetadata)[[source]](../_modules/torchcodec/decoders/_image_decoders.html#get_image_metadata)

Get the metadata of an encoded image, without decoding it.

The format is detected from the encoded data, like in
[`decode_image()`](torchcodec.decoders.decode_image.html#torchcodec.decoders.decode_image).

See [`ImageMetadata`](torchcodec.decoders.ImageMetadata.html#torchcodec.decoders.ImageMetadata) for the returned attributes, which all describe
the output of the decoders.

Example

```
from torchcodec.decoders import get_image_metadata

metadata = get_image_metadata("image.jpg")
print(metadata.width, metadata.height)
```

Parameters:

**source** (str, `pathlib.Path`, bytes, or `torch.Tensor`) - The encoded image data: a path (`str` or `pathlib.Path`), a
`bytes` object, or a 1-D uint8 `torch.Tensor` of the raw encoded
bytes.

Returns:

The image metadata.

Return type:

[ImageMetadata](torchcodec.decoders.ImageMetadata.html#torchcodec.decoders.ImageMetadata)