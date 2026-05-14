# Resize

*class*torchcodec.transforms.Resize(*size: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[int](https://docs.python.org/3/library/functions.html#int)]*)[[source]](../_modules/torchcodec/transforms/_decoder_transforms.html#Resize)

Resize the decoded frame to a given size.

Complementary TorchVision transform: [`Resize`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Resize.html#torchvision.transforms.v2.Resize).
Interpolation is always bilinear. Anti-aliasing is always on.

Parameters:

**size** (*Sequence**[*[*int*](https://docs.python.org/3/library/functions.html#int)*]*) - Desired output size. Must be a sequence of
the form (height, width).

Examples using `Resize`:

![](../_images/sphx_glr_transforms_thumb.png)

[Decoder Transforms: Applying transforms during decoding](../generated_examples/decoding/transforms.html)

Decoder Transforms: Applying transforms during decoding