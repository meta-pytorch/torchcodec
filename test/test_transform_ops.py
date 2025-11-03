# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import json
import os
import subprocess

import pytest

import torch

from torchcodec._core import (
    _add_video_stream,
    add_video_stream,
    create_from_file,
    get_frame_at_index,
    get_json_metadata,
)

from torchvision.transforms import v2

from .utils import (
    assert_frames_equal,
    assert_tensor_close_on_at_least,
    AV1_VIDEO,
    H265_VIDEO,
    NASA_VIDEO,
    needs_cuda,
)

torch._dynamo.config.capture_dynamic_output_shape_ops = True


class TestCoreVideoDecoderTransformOps:
    @pytest.mark.parametrize("video", [NASA_VIDEO, H265_VIDEO, AV1_VIDEO])
    def test_color_conversion_library(self, video):
        decoder = create_from_file(str(video.path))
        add_video_stream(decoder)
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        num_frames = metadata_dict["numFramesFromHeader"]

        filtergraph_decoder = create_from_file(str(video.path))
        _add_video_stream(
            filtergraph_decoder,
            color_conversion_library="filtergraph",
        )

        swscale_decoder = create_from_file(str(video.path))
        _add_video_stream(
            swscale_decoder,
            color_conversion_library="swscale",
        )

        for frame_index in [
            0,
            int(num_frames * 0.25),
            int(num_frames * 0.5),
            int(num_frames * 0.75),
            num_frames - 1,
        ]:
            filtergraph_frame, *_ = get_frame_at_index(
                filtergraph_decoder, frame_index=frame_index
            )
            swscale_frame, *_ = get_frame_at_index(
                swscale_decoder, frame_index=frame_index
            )

            assert_frames_equal(filtergraph_frame, swscale_frame)

    @pytest.mark.parametrize("width", [30, 32, 300])
    @pytest.mark.parametrize("height", [128])
    def test_color_conversion_library_with_generated_videos(
        self, tmp_path, width, height
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

        num_frames = metadata_dict["numFramesFromHeader"]
        assert num_frames is not None and num_frames == 1

        filtergraph_decoder = create_from_file(str(video_path))
        _add_video_stream(
            filtergraph_decoder,
            color_conversion_library="filtergraph",
        )

        auto_decoder = create_from_file(str(video_path))
        _add_video_stream(
            auto_decoder,
            color_conversion_library="swscale",
        )

        filtergraph_frame0, *_ = get_frame_at_index(filtergraph_decoder, frame_index=0)
        swscale_frame0, *_ = get_frame_at_index(auto_decoder, frame_index=0)
        assert_frames_equal(filtergraph_frame0, swscale_frame0)

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

    @pytest.mark.parametrize(
        "height_scaling_factor, width_scaling_factor",
        ((1.5, 1.31), (0.5, 0.71), (0.7, 1.31), (1.5, 0.71), (1.0, 1.0), (2.0, 2.0)),
    )
    def test_resize_torchvision(self, height_scaling_factor, width_scaling_factor):
        height = int(NASA_VIDEO.get_height() * height_scaling_factor)
        width = int(NASA_VIDEO.get_width() * width_scaling_factor)
        resize_spec = f"resize, {height}, {width}"

        decoder_resize = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder_resize, transform_specs=resize_spec)

        decoder_full = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder_full)

        for frame_index in [0, 10, 17, 100, 230, 389]:
            expected_shape = (NASA_VIDEO.get_num_color_channels(), height, width)
            frame_resize, *_ = get_frame_at_index(
                decoder_resize, frame_index=frame_index
            )

            frame_full, *_ = get_frame_at_index(decoder_full, frame_index=frame_index)
            frame_tv = v2.functional.resize(frame_full, size=(height, width))

            assert frame_resize.shape == expected_shape
            assert frame_tv.shape == expected_shape

            assert_tensor_close_on_at_least(
                frame_resize, frame_tv, percentage=99, atol=1
            )

    def test_resize_ffmpeg(self):
        height = 135
        width = 240
        expected_shape = (NASA_VIDEO.get_num_color_channels(), height, width)
        resize_spec = f"resize, {height}, {width}"
        resize_filtergraph = f"scale={width}:{height}:flags=bilinear"

        decoder_resize = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder_resize, transform_specs=resize_spec)

        for frame_index in [17, 230, 389]:
            frame_resize, *_ = get_frame_at_index(
                decoder_resize, frame_index=frame_index
            )
            frame_ref = NASA_VIDEO.get_frame_data_by_index(
                frame_index, filters=resize_filtergraph
            )

            assert frame_resize.shape == expected_shape
            assert frame_ref.shape == expected_shape
            assert_frames_equal(frame_resize, frame_ref)

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
        width = 300
        height = 200
        x = 50
        y = 35
        crop_spec = f"crop, {height}, {width}, {x}, {y}"
        crop_filtergraph = f"crop={width}:{height}:{x}:{y}:exact=1"
        expected_shape = (NASA_VIDEO.get_num_color_channels(), height, width)

        decoder_crop = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder_crop, transform_specs=crop_spec)

        decoder_full = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder_full)

        for frame_index in [0, 15, 200, 389]:
            frame_crop, *_ = get_frame_at_index(decoder_crop, frame_index=frame_index)
            frame_ref = NASA_VIDEO.get_frame_data_by_index(
                frame_index, filters=crop_filtergraph
            )

            frame_full, *_ = get_frame_at_index(decoder_full, frame_index=frame_index)
            frame_tv = v2.functional.crop(
                frame_full, top=y, left=x, height=height, width=width
            )

            assert frame_crop.shape == expected_shape
            assert frame_ref.shape == expected_shape
            assert frame_tv.shape == expected_shape

            assert_frames_equal(frame_crop, frame_ref)
            assert_frames_equal(frame_crop, frame_tv)

    def test_crop_transform_fails(self):

        with pytest.raises(
            RuntimeError,
            match="must have 5 elements",
        ):
            decoder = create_from_file(str(NASA_VIDEO.path))
            add_video_stream(decoder, transform_specs="crop, 100, 100")

        with pytest.raises(
            RuntimeError,
            match="must be a positive integer",
        ):
            decoder = create_from_file(str(NASA_VIDEO.path))
            add_video_stream(decoder, transform_specs="crop, -10, 100, 100, 100")

        with pytest.raises(
            RuntimeError,
            match="cannot be converted to an int",
        ):
            decoder = create_from_file(str(NASA_VIDEO.path))
            add_video_stream(decoder, transform_specs="crop, 100, 100, blah, 100")

        with pytest.raises(
            RuntimeError,
            match="x position out of bounds",
        ):
            decoder = create_from_file(str(NASA_VIDEO.path))
            add_video_stream(decoder, transform_specs="crop, 100, 100, 9999, 100")

        with pytest.raises(
            RuntimeError,
            match="y position out of bounds",
        ):
            decoder = create_from_file(str(NASA_VIDEO.path))
            add_video_stream(decoder, transform_specs="crop, 999, 100, 100, 100")
