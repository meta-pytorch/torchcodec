# DecoderTransform

*class*torchcodec.transforms.DecoderTransform[[source]](../_modules/torchcodec/transforms/_decoder_transforms.html#DecoderTransform)

Base class for all decoder transforms.

A *decoder transform* is a transform that is applied by the decoder before
returning the decoded frame. Applying decoder transforms to frames
should be both faster and more memory efficient than receiving normally
decoded frames and applying the same kind of transform.

Most `DecoderTransform` objects have a complementary transform in TorchVision,
specificially in [torchvision.transforms.v2](https://docs.pytorch.org/vision/stable/transforms.html).
For such transforms, we ensure that:

> 1. The names are the same.
> 2. Default behaviors are the same.
> 3. The parameters for the `DecoderTransform` object are a subset of the
> TorchVision [`Transform`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Transform.html#torchvision.transforms.v2.Transform) object.
> 4. Parameters with the same name control the same behavior and accept a
> subset of the same types.
> 5. The difference between the frames returned by a decoder transform and
> the complementary TorchVision transform are such that a model should
> not be able to tell the difference.

Examples using `DecoderTransform`:

![](../_images/sphx_glr_performance_tips_thumb.jpg)

[TorchCodec Performance Tips and Best Practices](../generated_examples/decoding/performance_tips.html)

TorchCodec Performance Tips and Best Practices
![](../_images/sphx_glr_transforms_thumb.png)

[Decoder Transforms: Applying transforms during decoding](../generated_examples/decoding/transforms.html)

Decoder Transforms: Applying transforms during decoding