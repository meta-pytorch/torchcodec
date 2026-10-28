# ImageMetadata

*class*torchcodec.decoders.ImageMetadata(*format: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*, *width: [int](https://docs.python.org/3/builtins/functions.html#int)*, *height: [int](https://docs.python.org/3/builtins/functions.html#int)*, *num_channels: [int](https://docs.python.org/3/builtins/functions.html#int)*, *bit_depth: [int](https://docs.python.org/3/builtins/functions.html#int)*, *num_frames: [int](https://docs.python.org/3/builtins/functions.html#int)*, *rotation: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*)[[source]](../_modules/torchcodec/decoders/_image_decoders.html#ImageMetadata)

Metadata of an encoded image, as returned by [`get_image_metadata()`](torchcodec.decoders.get_image_metadata.html#torchcodec.decoders.get_image_metadata).

Every attribute describes the tensor that [`decode_image()`](torchcodec.decoders.decode_image.html#torchcodec.decoders.decode_image) (or the
matching format-specific decoder) returns for this image.

bit_depth*: [int](https://docs.python.org/3/builtins/functions.html#int)*

Number of bits per channel in the encoded image. When greater than 8,
decoding with `output_dtype="auto"` returns a `torch.uint16` tensor.

format*: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The image format, e.g. `"jpeg"`.

height*: [int](https://docs.python.org/3/builtins/functions.html#int)*

Height of the decoded image, **post-rotation**. This is the height of the
tensor returned by [`decode_image()`](torchcodec.decoders.decode_image.html#torchcodec.decoders.decode_image) or the format-specific decoders,
which always apply the EXIF orientation.

num_channels*: [int](https://docs.python.org/3/builtins/functions.html#int)*

Number of channels of the image. This is the number of channels
of the decoded image when decoding with `mode="UNCHANGED"`.

num_frames*: [int](https://docs.python.org/3/builtins/functions.html#int)*

1 for still images, and the
size of the leading `N` dimension of the decoded tensor for animated or
multi-image files.

Type:

Number of frames (or images) in the file

rotation*: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

one of 0, 90, -90 or 180)
from the orientation metadata (EXIF orientation, or the AVIF rotation and
mirroring properties), or None if the image has no orientation metadata. The
decoders always apply it, so the returned images are in the correct
orientation.

The orientation metadata can also specify a mirroring, which the decoders
apply too but isn't reported here, since it doesn't change the image
dimensions.

Type:

Rotation angle in degrees (counter-clockwise

width*: [int](https://docs.python.org/3/builtins/functions.html#int)*

Width of the decoded image, **post-rotation**. This is the width of the
tensor returned by [`decode_image()`](torchcodec.decoders.decode_image.html#torchcodec.decoders.decode_image) or the format-specific decoders,
which always apply the EXIF orientation.