# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os

os.environ["TORCH_LOGS"] = "output_code"
import json
import subprocess

import pytest

import torch

from torchcodec._core import (
    _add_video_stream,
    add_video_stream,
    create_from_file,
    get_frame_at_index,
    get_json_metadata,
    get_next_frame,
)

from torchvision.transforms import v2

from .utils import assert_frames_equal, NASA_VIDEO, needs_cuda

torch._dynamo.config.capture_dynamic_output_shape_ops = True


class TestVideoDecoderTransformOps:
    # We choose arbitrary values for width and height scaling to get better
    # test coverage. Some pairs upscale the image while others downscale it.
    @pytest.mark.parametrize(
        "width_scaling_factor,height_scaling_factor",
        ((1.31, 1.5), (0.71, 0.5), (1.31, 0.7), (0.71, 1.5), (1.0, 1.0)),
    )
    @pytest.mark.parametrize("input_video", [NASA_VIDEO])
    def test_color_conversion_library_with_scaling(
        self, input_video, width_scaling_factor, height_scaling_factor
    ):
        decoder = create_from_file(str(input_video.path))
        add_video_stream(decoder)
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        assert metadata_dict["width"] == input_video.width
        assert metadata_dict["height"] == input_video.height

        target_height = int(input_video.height * height_scaling_factor)
        target_width = int(input_video.width * width_scaling_factor)
        if width_scaling_factor != 1.0:
            assert target_width != input_video.width
        if height_scaling_factor != 1.0:
            assert target_height != input_video.height

        filtergraph_decoder = create_from_file(str(input_video.path))
        _add_video_stream(
            filtergraph_decoder,
            transform_specs=f"resize, {target_height}, {target_width}",
            color_conversion_library="filtergraph",
        )
        filtergraph_frame0, _, _ = get_next_frame(filtergraph_decoder)

        swscale_decoder = create_from_file(str(input_video.path))
        _add_video_stream(
            swscale_decoder,
            transform_specs=f"resize, {target_height}, {target_width}",
            color_conversion_library="swscale",
        )
        swscale_frame0, _, _ = get_next_frame(swscale_decoder)
        assert_frames_equal(filtergraph_frame0, swscale_frame0)
        assert filtergraph_frame0.shape == (3, target_height, target_width)

    @pytest.mark.parametrize(
        "width_scaling_factor,height_scaling_factor",
        ((1.31, 1.5), (0.71, 0.5), (1.31, 0.7), (0.71, 1.5), (1.0, 1.0)),
    )
    @pytest.mark.parametrize("width", [30, 32, 300])
    @pytest.mark.parametrize("height", [128])
    def test_color_conversion_library_with_generated_videos(
        self, tmp_path, width, height, width_scaling_factor, height_scaling_factor
    ):
        # We consider filtergraph to be the reference color conversion library.
        # However the video decoder sometimes uses swscale as that is faster.
        # The exact color conversion library used is an implementation detail
        # of the video decoder and depends on the video's width.
        #
        # In this test we compare the output of filtergraph (which is the
        # reference) with the output of the video decoder (which may use
        # swscale if it chooses for certain video widths) to make sure they are
        # always the same.
        video_path = f"{tmp_path}/frame_numbers_{width}x{height}.mp4"
        # We don't specify a particular encoder because the ffmpeg binary could
        # be configured with different encoders. For the purposes of this test,
        # the actual encoder is irrelevant.
        with contextlib.ExitStack() as stack:
            ffmpeg_cli = "ffmpeg"

            if os.environ.get("IN_FBCODE_TORCHCODEC") == "1":
                import importlib.resources

                ffmpeg_cli = stack.enter_context(
                    importlib.resources.path(__package__, "ffmpeg")
                )

            command = [
                ffmpeg_cli,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=blue",
                "-pix_fmt",
                "yuv420p",
                "-s",
                f"{width}x{height}",
                "-frames:v",
                "1",
                video_path,
            ]
            subprocess.check_call(command)

        decoder = create_from_file(str(video_path))
        add_video_stream(decoder)
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        assert metadata_dict["width"] == width
        assert metadata_dict["height"] == height

        target_height = int(height * height_scaling_factor)
        target_width = int(width * width_scaling_factor)
        if width_scaling_factor != 1.0:
            assert target_width != width
        if height_scaling_factor != 1.0:
            assert target_height != height

        filtergraph_decoder = create_from_file(str(video_path))
        _add_video_stream(
            filtergraph_decoder,
            transform_specs=f"resize, {target_height}, {target_width}",
            color_conversion_library="filtergraph",
        )
        filtergraph_frame0, _, _ = get_next_frame(filtergraph_decoder)

        auto_decoder = create_from_file(str(video_path))
        add_video_stream(
            auto_decoder,
            transform_specs=f"resize, {target_height}, {target_width}",
        )
        auto_frame0, _, _ = get_next_frame(auto_decoder)
        assert_frames_equal(filtergraph_frame0, auto_frame0)

    @needs_cuda
    def test_scaling_on_cuda_fails(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        with pytest.raises(
            RuntimeError,
            match="Transforms are only supported for CPU devices.",
        ):
            add_video_stream(decoder, device="cuda", transform_specs="resize, 100, 100")

    def test_transform_fails(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        with pytest.raises(
            RuntimeError,
            match="Invalid transform spec",
        ):
            add_video_stream(decoder, transform_specs=";")

        with pytest.raises(
            RuntimeError,
            match="Invalid transform name",
        ):
            add_video_stream(decoder, transform_specs="invalid, 1, 2")

    def test_resize_transform_fails(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        with pytest.raises(
            RuntimeError,
            match="must have 3 elements",
        ):
            add_video_stream(decoder, transform_specs="resize, 100, 100, 100")

        with pytest.raises(
            RuntimeError,
            match="must be a positive integer",
        ):
            add_video_stream(decoder, transform_specs="resize, -10, 100")

        with pytest.raises(
            RuntimeError,
            match="must be a positive integer",
        ):
            add_video_stream(decoder, transform_specs="resize, 100, 0")

        with pytest.raises(
            RuntimeError,
            match="cannot be converted to an int",
        ):
            add_video_stream(decoder, transform_specs="resize, blah, 100")

        with pytest.raises(
            RuntimeError,
            match="out of range",
        ):
            add_video_stream(decoder, transform_specs="resize, 100, 1000000000000")

    def test_crop_transform(self):
        # Note that filtergraph accepts dimensions as (w, h) and we accept them as (h, w).
        crop_spec = "crop, 200, 300, 50, 35"  # h=200, w=300, x=50, y=35
        crop_filtergraph = "crop=300:200:50:35:exact=1"  # w=300, h=200, x=50, y=35
        expected_shape = (3, 200, 300)  # channels=3, height=200, width=300

        decoder_crop = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder_crop, transform_specs=crop_spec)

        decoder_full = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder_full)

        for frame_index in [0, 15, 200, 389]:
            frame, *_ = get_frame_at_index(decoder_crop, frame_index=frame_index)
            frame_ref = NASA_VIDEO.get_frame_data_by_index(
                frame_index, filters=crop_filtergraph
            )

            frame_full, *_ = get_frame_at_index(decoder_full, frame_index=frame_index)
            frame_tv = v2.functional.crop(
                frame_full, top=35, left=50, height=200, width=300
            )

            assert frame.shape == expected_shape
            assert frame_ref.shape == expected_shape
            assert frame_tv.shape == expected_shape

            assert_frames_equal(frame, frame_tv)
            assert_frames_equal(frame, frame_ref)

    def test_crop_transform_fails(self):
        decoder = create_from_file(str(NASA_VIDEO.path))

        with pytest.raises(
            RuntimeError,
            match="must have 5 elements",
        ):
            add_video_stream(decoder, transform_specs="crop, 100, 100")

        with pytest.raises(
            RuntimeError,
            match="must be a positive integer",
        ):
            add_video_stream(decoder, transform_specs="crop, -10, 100, 100, 100")

        with pytest.raises(
            RuntimeError,
            match="cannot be converted to an int",
        ):
            add_video_stream(decoder, transform_specs="crop, 100, 100, blah, 100")

        with pytest.raises(
            RuntimeError,
            match="x position out of bounds",
        ):
            add_video_stream(decoder, transform_specs="crop, 100, 100, 9999, 100")
