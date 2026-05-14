# CenterCrop

*class*torchcodec.transforms.CenterCrop(*size: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[int](https://docs.python.org/3/library/functions.html#int)]*)[[source]](../_modules/torchcodec/transforms/_decoder_transforms.html#CenterCrop)

Crop the decoded frame to a given size in the center of the frame.

Complementary TorchVision transform: [`CenterCrop`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.CenterCrop.html#torchvision.transforms.v2.CenterCrop).

Parameters:

**size** (*Sequence**[*[*int*](https://docs.python.org/3/library/functions.html#int)*]*) - Desired output size. Must be a sequence of
the form (height, width).

Examples using `CenterCrop`:

![](../_images/sphx_glr_transforms_thumb.png)

[Decoder Transforms: Applying transforms during decoding](../generated_examples/decoding/transforms.html)

Decoder Transforms: Applying transforms during decoding