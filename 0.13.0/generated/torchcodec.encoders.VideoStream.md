# VideoStream

*class*torchcodec.encoders.VideoStream(*encoder_tensor: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *stream_index: [int](https://docs.python.org/3/library/functions.html#int)*)[[source]](../_modules/torchcodec/encoders/_multi_stream_encoder.html#VideoStream)

A video stream within an [`Encoder`](torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder).

Returned by [`Encoder.add_video()`](torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder.add_video). Use `add_frames()` to feed
video frames into this stream.

Examples using `VideoStream`:

![](../_images/sphx_glr_multi_stream_encoding_thumb.jpg)

[Encoding audio and video streams with the Encoder](../generated_examples/encoding/multi_stream_encoding.html)

Encoding audio and video streams with the Encoder
![](../_images/sphx_glr_video_encoding_thumb.jpg)

[Encoding video with the Encoder](../generated_examples/encoding/video_encoding.html)

Encoding video with the Encoder

add_frames(*frames: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torchcodec/encoders/_multi_stream_encoder.html#VideoStream.add_frames)

Add video frames to this stream.

Parameters:

**frames** (`torch.Tensor`) - The frames to encode. This must be a 4D
tensor of shape `(N, C, H, W)` where N is the number of
frames, C is 3 channels (RGB), H is height, and W is width.
Values must be uint8 in the range `[0, 255]`. The device of
the tensor must match the `device` passed to
[`Encoder.add_video()`](torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder.add_video).