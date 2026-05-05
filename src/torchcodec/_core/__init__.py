# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from ._metadata import (
    AudioStreamMetadata,
    ContainerMetadata,
    get_container_metadata,
    get_container_metadata_from_header,
    VideoStreamMetadata,
)
from .ops import (
    _get_backend_details,
    _get_key_frame_indices,
    _get_nvdec_cache_size,
    _test_frame_pts_equality,
    core_library_path,
    create_streaming_encoder_to_file,
    create_streaming_encoder_to_file_like,
    create_wav_decoder_from_file,
    encode_audio_to_file,
    encode_audio_to_file_like,
    encode_audio_to_tensor,
    encode_video_to_file,
    encode_video_to_file_like,
    encode_video_to_tensor,
    ffmpeg_major_version,
    get_ffmpeg_library_versions,
    get_frame_at_index,
    get_frame_at_pts,
    get_frames_at_indices,
    get_frames_by_pts,
    get_frames_by_pts_in_range,
    get_frames_by_pts_in_range_audio,
    get_frames_in_range,
    get_json_metadata,
    get_next_frame,
    get_nvdec_cache_capacity,
    get_wav_metadata_from_decoder,
    get_wav_samples_in_range,
    set_nvdec_cache_capacity,
    streaming_encoder_add_audio_stream,
    streaming_encoder_add_frames,
    streaming_encoder_add_video_stream,
    streaming_encoder_close,
    streaming_encoder_open,
)
