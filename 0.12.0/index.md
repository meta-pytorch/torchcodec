# Welcome to the TorchCodec documentation!

TorchCodec is a Python library for decoding video and audio data into PyTorch
tensors, on CPU and CUDA GPU. It also supports audio and video encoding!
It aims to be fast, easy to use, and well integrated into the PyTorch ecosystem.
If you want to use PyTorch to train ML models on videos and audio, TorchCodec is
how you turn these into data.

We achieve these capabilities through:

- Pythonic APIs that mirror Python and PyTorch conventions.
- Relying on [FFmpeg](https://www.ffmpeg.org/) to do the decoding / encoding.
TorchCodec uses the version of FFmpeg you already have installed. FFmpeg is a
mature library with broad coverage available on most systems. It is, however,
not easy to use. TorchCodec abstracts FFmpeg's complexity to ensure it is
used correctly and efficiently.
- Returning data as PyTorch tensors, ready to be fed into PyTorch transforms
or used directly to train models.

## Installation instructions

Installation instructions

How to install TorchCodec

[https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec](https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec)

## Decoding

Getting Started with TorchCodec

A simple video decoding example

[generated_examples/decoding/basic_example.html](generated_examples/decoding/basic_example.html)

Audio Decoding

A simple audio decoding example

[generated_examples/decoding/audio_decoding.html](generated_examples/decoding/audio_decoding.html)

GPU decoding

A simple example demonstrating CUDA GPU decoding

[generated_examples/decoding/basic_cuda_example.html](generated_examples/decoding/basic_cuda_example.html)

Streaming video

How to efficiently decode videos from the cloud

[generated_examples/decoding/file_like.html](generated_examples/decoding/file_like.html)

Parallel decoding

How to decode a video with multiple processes or threads.

[generated_examples/decoding/parallel_decoding.html](generated_examples/decoding/parallel_decoding.html)

Clip sampling

How to sample regular and random clips from a video

[generated_examples/decoding/sampling.html](generated_examples/decoding/sampling.html)

Decoder transforms

How to apply transforms while decoding

[generated_examples/decoding/transforms.html](generated_examples/decoding/transforms.html)

Performance Tips

Tips for optimizing video decoding performance

[generated_examples/decoding/performance_tips.html](generated_examples/decoding/performance_tips.html)

## Encoding

Audio Encoding

How encode audio samples

[generated_examples/encoding/audio_encoding.html](generated_examples/encoding/audio_encoding.html)

Video Encoding

How to encode video frames

[generated_examples/encoding/video_encoding.html](generated_examples/encoding/video_encoding.html)