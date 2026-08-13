# RandomCrop

*class*torchcodec.transforms.RandomCrop(*size: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[int](https://docs.python.org/3/library/functions.html#int)]*)[[source]](../_modules/torchcodec/transforms/_decoder_transforms.html#RandomCrop)

Crop the decoded frame to a given size at a random location in the frame.

Complementary TorchVision transform: [`RandomCrop`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomCrop.html#torchvision.transforms.v2.RandomCrop).
Padding of all kinds is disabled. The random location within the frame is
determined during the initialization of the
[`VideoDecoder`](torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) object that owns this transform.
As a consequence, each decoded frame in the video will be cropped at the
same location. Videos with variable resolution may result in undefined
behavior.

Parameters:

**size** (*Sequence**[*[*int*](https://docs.python.org/3/library/functions.html#int)*]*) - Desired output size. Must be a sequence of
the form (height, width).