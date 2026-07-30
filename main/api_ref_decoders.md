# torchcodec.decoders

## Video Decoding

For a video decoder tutorial, see: [Decoding a video with VideoDecoder](generated_examples/decoding/basic_example.html#sphx-glr-generated-examples-decoding-basic-example-py).

| [`VideoDecoder`](generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) | A single-stream video decoder. |
| --- | --- |
| [`WavDecoder`](generated/torchcodec.decoders.WavDecoder.html#torchcodec.decoders.WavDecoder) | A fast decoder for WAV audio files. |

| [`VideoStreamMetadata`](generated/torchcodec.decoders.VideoStreamMetadata.html#torchcodec.decoders.VideoStreamMetadata) | Metadata of a single video stream. |
| --- | --- |

**CUDA decoding utils:**

| [`set_cuda_backend`](generated/torchcodec.decoders.set_cuda_backend.html#torchcodec.decoders.set_cuda_backend) | Context Manager to set the CUDA backend for [`VideoDecoder`](generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder). |
| --- | --- |
| [`set_nvdec_cache_capacity`](generated/torchcodec.decoders.set_nvdec_cache_capacity.html#torchcodec.decoders.set_nvdec_cache_capacity) | Set the maximum number of NVDEC decoders that can be cached (per GPU). |
| [`get_nvdec_cache_capacity`](generated/torchcodec.decoders.get_nvdec_cache_capacity.html#torchcodec.decoders.get_nvdec_cache_capacity) | Get the capacity of the per-device NVDEC decoder cache. |

| [`CpuFallbackStatus`](generated/torchcodec.decoders.CpuFallbackStatus.html#torchcodec.decoders.CpuFallbackStatus) | Information about CPU fallback status. |
| --- | --- |

## Audio Decoding

For an audio decoder tutorial, see: [Decoding audio streams with AudioDecoder](generated_examples/decoding/audio_decoding.html#sphx-glr-generated-examples-decoding-audio-decoding-py).

| [`AudioDecoder`](generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) | A single-stream audio decoder. |
| --- | --- |
| [`WavDecoder`](generated/torchcodec.decoders.WavDecoder.html#torchcodec.decoders.WavDecoder) | A fast decoder for WAV audio files. |

| [`AudioStreamMetadata`](generated/torchcodec.decoders.AudioStreamMetadata.html#torchcodec.decoders.AudioStreamMetadata) | Metadata of a single audio stream. |
| --- | --- |

## Image Decoding

| [`decode_image`](generated/torchcodec.decoders.decode_image.html#torchcodec.decoders.decode_image) | Decode an image into a `[N]CHW` tensor, detecting the format automatically. |
| --- | --- |
| [`decode_jpeg`](generated/torchcodec.decoders.decode_jpeg.html#torchcodec.decoders.decode_jpeg) | Decode a JPEG image into a `CHW` tensor, on CPU or CUDA. |
| [`decode_png`](generated/torchcodec.decoders.decode_png.html#torchcodec.decoders.decode_png) | Decode a PNG image into a `CHW` tensor. |
| [`decode_webp`](generated/torchcodec.decoders.decode_webp.html#torchcodec.decoders.decode_webp) | Decode a WebP image into a `[N]CHW` tensor. |
| [`decode_gif`](generated/torchcodec.decoders.decode_gif.html#torchcodec.decoders.decode_gif) | Decode a GIF image into a `[N]CHW` tensor. |
| [`decode_avif`](generated/torchcodec.decoders.decode_avif.html#torchcodec.decoders.decode_avif) | Decode an AVIF image into a `[N]CHW` tensor. |
| [`decode_heic`](generated/torchcodec.decoders.decode_heic.html#torchcodec.decoders.decode_heic) | Decode an HEIC/HEIF image into a `[N]CHW` tensor - requires `libheif`! |

| [`ImageReadMode`](generated/torchcodec.decoders.ImageReadMode.html#torchcodec.decoders.ImageReadMode) | Color mode for image decoding. |
| --- | --- |