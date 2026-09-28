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

## Low-level decoding APIs

Important

**The low-level APIs are in beta.** Their signatures and semantics may still
change slightly, in response to user feedback.

These expose the three stages of decoding - demuxing, decoding and conversion -
as separate objects, where [`VideoDecoder`](generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) and [`AudioDecoder`](generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) do
all three for you. Reach for them when you need to control a stage, skip one, or
run them on different threads. If you just want frames out of a file, use
[`VideoDecoder`](generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder).

For tutorials, see:

- [Build your own decoding pipeline](generated_examples/low_level/basics.html#sphx-glr-generated-examples-low-level-basics-py)
- [Multi-threaded decoding pipelines](generated_examples/low_level/pipelines.html#sphx-glr-generated-examples-low-level-pipelines-py)
- [Raw frames and raw audio samples](generated_examples/low_level/raw_data.html#sphx-glr-generated-examples-low-level-raw-data-py)
- [CUDA streams](generated_examples/low_level/cuda_streams.html#sphx-glr-generated-examples-low-level-cuda-streams-py)

### Demuxing

| [`Demuxer`](generated/torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer) | Reads one or more video and audio streams from a container, and produces their compressed [`Packet`](generated/torchcodec.decoders.Packet.html#torchcodec.decoders.Packet)s. |
| --- | --- |
| [`VideoStream`](generated/torchcodec.decoders.VideoStream.html#torchcodec.decoders.VideoStream) | A video stream followed by a [`Demuxer`](generated/torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer). |
| [`AudioStream`](generated/torchcodec.decoders.AudioStream.html#torchcodec.decoders.AudioStream) | An audio stream followed by a [`Demuxer`](generated/torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer). |

| [`get_container_metadata`](generated/torchcodec.decoders.get_container_metadata.html#torchcodec.decoders.get_container_metadata) | Describe a container and every stream in it, without decoding anything. |
| --- | --- |

### Decoding

| [`VideoPacketDecoder`](generated/torchcodec.decoders.VideoPacketDecoder.html#torchcodec.decoders.VideoPacketDecoder) | Decodes the compressed [`Packet`](generated/torchcodec.decoders.Packet.html#torchcodec.decoders.Packet)s of one video stream into [`RawFrame`](generated/torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame)s. |
| --- | --- |
| [`AudioPacketDecoder`](generated/torchcodec.decoders.AudioPacketDecoder.html#torchcodec.decoders.AudioPacketDecoder) | Decodes the compressed [`Packet`](generated/torchcodec.decoders.Packet.html#torchcodec.decoders.Packet)s of one audio stream into [`RawAudioSamples`](generated/torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples). |

### Conversion

| [`ColorConverter`](generated/torchcodec.decoders.ColorConverter.html#torchcodec.decoders.ColorConverter) | Turn a [`RawFrame`](generated/torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame) (typically YUV) into an RGB [`Frame`](generated/torchcodec.Frame.html#torchcodec.Frame). |
| --- | --- |
| [`AudioConverter`](generated/torchcodec.decoders.AudioConverter.html#torchcodec.decoders.AudioConverter) | Turn [`RawAudioSamples`](generated/torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples) into normalised float32 [`AudioSamples`](generated/torchcodec.AudioSamples.html#torchcodec.AudioSamples), optionally resampling and remixing channels. |

### Data types

| [`Packet`](generated/torchcodec.decoders.Packet.html#torchcodec.decoders.Packet) | One compressed packet of one stream, as a [`Demuxer`](generated/torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer) produced it. |
| --- | --- |
| [`RawFrame`](generated/torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame) | One decoded video frame, exactly as the decoder produced it. |

| [`FrameIndex`](generated/torchcodec.decoders.FrameIndex.html#torchcodec.decoders.FrameIndex) | What the packets of a video stream say about its frames, as returned by a [scan](glossary.html#term-scan) ([`VideoStream.scan()`](generated/torchcodec.decoders.VideoStream.html#torchcodec.decoders.VideoStream.scan)). |
| --- | --- |
| [`RawAudioSamples`](generated/torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples) | One decoded audio frame's samples, exactly as the decoder produced them. |

### Metadata

| [`DemuxerMetadata`](generated/torchcodec.decoders.DemuxerMetadata.html#torchcodec.decoders.DemuxerMetadata) | Container-level metadata, as reported by the header. |
| --- | --- |
| [`ContainerMetadata`](generated/torchcodec.decoders.ContainerMetadata.html#torchcodec.decoders.ContainerMetadata) | Metadata of a container and of every stream in it. |
| [`VideoStreamHeaderMetadata`](generated/torchcodec.decoders.VideoStreamHeaderMetadata.html#torchcodec.decoders.VideoStreamHeaderMetadata) | Metadata of a single video stream, as reported by the container header. |
| [`AudioStreamHeaderMetadata`](generated/torchcodec.decoders.AudioStreamHeaderMetadata.html#torchcodec.decoders.AudioStreamHeaderMetadata) | Metadata of a single audio stream, as reported by the container header. |
| [`StreamMetadata`](generated/torchcodec.decoders.StreamMetadata.html#torchcodec.decoders.StreamMetadata) | Metadata of a single stream, as reported by the container header. |