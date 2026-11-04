# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# `x as x` marks public re-exports for type-checkers, see torchcodec/__init__.py.
from .._core import (
    AudioStreamMetadata as AudioStreamMetadata,
    VideoStreamMetadata as VideoStreamMetadata,
)
from ._audio_decoder import AudioDecoder as AudioDecoder
from ._blocks import (
    AudioConverter as AudioConverter,
    AudioPacketDecoder as AudioPacketDecoder,
    AudioStream as AudioStream,
    AudioStreamHeaderMetadata as AudioStreamHeaderMetadata,
    ColorConverter as ColorConverter,
    ContainerMetadata as ContainerMetadata,
    Demuxer as Demuxer,
    DemuxerMetadata as DemuxerMetadata,
    FrameIndex as FrameIndex,
    get_container_metadata as get_container_metadata,
    Packet as Packet,
    RawAudioSamples as RawAudioSamples,
    RawFrame as RawFrame,
    StreamMetadata as StreamMetadata,
    VideoPacketDecoder as VideoPacketDecoder,
    VideoStream as VideoStream,
    VideoStreamHeaderMetadata as VideoStreamHeaderMetadata,
)
from ._decoder_utils import (
    get_nvdec_cache_capacity as get_nvdec_cache_capacity,
    set_cuda_backend as set_cuda_backend,
    set_nvdec_cache_capacity as set_nvdec_cache_capacity,
)
from ._image_decoders import (  # noqa
    decode_avif as decode_avif,
    decode_gif as decode_gif,
    decode_heic as decode_heic,
    decode_image as decode_image,
    decode_jpeg as decode_jpeg,
    decode_png as decode_png,
    decode_webp as decode_webp,
    get_image_metadata as get_image_metadata,
    ImageMetadata as ImageMetadata,
    ImageReadMode as ImageReadMode,
)
from ._video_decoder import (
    CpuFallbackStatus as CpuFallbackStatus,
    VideoDecoder as VideoDecoder,
)
from ._wav_decoder import WavDecoder as WavDecoder

SimpleVideoDecoder = VideoDecoder
