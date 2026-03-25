# torchcodec.decoders

For a video decoder tutorial, see: [Decoding a video with VideoDecoder](generated_examples/decoding/basic_example.html#sphx-glr-generated-examples-decoding-basic-example-py).
For an audio decoder tutorial, see: [Decoding audio streams with AudioDecoder](generated_examples/decoding/audio_decoding.html#sphx-glr-generated-examples-decoding-audio-decoding-py).

| [`VideoDecoder`](generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) | A single-stream video decoder. |
| --- | --- |
| [`AudioDecoder`](generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) | A single-stream audio decoder. |

| [`VideoStreamMetadata`](generated/torchcodec.decoders.VideoStreamMetadata.html#torchcodec.decoders.VideoStreamMetadata) | Metadata of a single video stream. |
| --- | --- |
| [`AudioStreamMetadata`](generated/torchcodec.decoders.AudioStreamMetadata.html#torchcodec.decoders.AudioStreamMetadata) | Metadata of a single audio stream. |

## CUDA decoding utils

| [`set_cuda_backend`](generated/torchcodec.decoders.set_cuda_backend.html#torchcodec.decoders.set_cuda_backend) | Context Manager to set the CUDA backend for [`VideoDecoder`](generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder). |
| --- | --- |
| [`set_nvdec_cache_capacity`](generated/torchcodec.decoders.set_nvdec_cache_capacity.html#torchcodec.decoders.set_nvdec_cache_capacity) | Set the maximum number of NVDEC decoders that can be cached (per GPU). |
| [`get_nvdec_cache_capacity`](generated/torchcodec.decoders.get_nvdec_cache_capacity.html#torchcodec.decoders.get_nvdec_cache_capacity) | Get the capacity of the per-device NVDEC decoder cache. |

| [`CpuFallbackStatus`](generated/torchcodec.decoders.CpuFallbackStatus.html#torchcodec.decoders.CpuFallbackStatus) | Information about CPU fallback status. |
| --- | --- |