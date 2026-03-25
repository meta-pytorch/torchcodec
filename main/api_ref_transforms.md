# torchcodec.transforms

For a tutorial, see: [Decoder Transforms: Applying transforms during decoding](generated_examples/decoding/transforms.html#sphx-glr-generated-examples-decoding-transforms-py).

| [`DecoderTransform`](generated/torchcodec.transforms.DecoderTransform.html#torchcodec.transforms.DecoderTransform) | Base class for all decoder transforms. |
| --- | --- |
| [`CenterCrop`](generated/torchcodec.transforms.CenterCrop.html#torchcodec.transforms.CenterCrop) | Crop the decoded frame to a given size in the center of the frame. |
| [`RandomCrop`](generated/torchcodec.transforms.RandomCrop.html#torchcodec.transforms.RandomCrop) | Crop the decoded frame to a given size at a random location in the frame. |
| [`Resize`](generated/torchcodec.transforms.Resize.html#torchcodec.transforms.Resize) | Resize the decoded frame to a given size. |