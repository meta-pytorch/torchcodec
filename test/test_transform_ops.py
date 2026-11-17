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
import torchcodec
from torchcodec._core import get_frame_at_index, get_json_metadata
from torchcodec._core.ops import _add_video_stream, add_video_stream, create_from_file
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import v2

from .utils import (
    assert_frames_equal,
    assert_tensor_close_on_at_least,
    AV1_VIDEO,
    get_ffmpeg_minor_version,
    H265_10BITS,
    H265_VIDEO,
    NASA_VIDEO,
    NASA_VIDEO_HDR,
    NASA_VIDEO_ROTATED,
    needs_cuda,
    TEST_NON_ZERO_START as NON_32_ALIGNED_WIDTH_VIDEO,
    TEST_SRC_2_720P,
)


class TestPublicVideoDecoderTransformOps:
    @pytest.mark.parametrize(
        "height_scaling_factor, width_scaling_factor",
        ((1.5, 1.31), (0.5, 0.71), (0.7, 1.31), (1.5, 0.71), (1.0, 1.0), (2.0, 2.0)),
    )
    @pytest.mark.parametrize("video", [NASA_VIDEO, TEST_SRC_2_720P])
    def test_resize_torchvision(
        self, video, height_scaling_factor, width_scaling_factor
    ):
        height = int(video.get_height() * height_scaling_factor)
        width = int(video.get_width() * width_scaling_factor)

        # We're using both the TorchCodec object and the TorchVision object to
        # ensure that they specify exactly the same thing.
        decoder_resize = VideoDecoder(
            video.path, transforms=[torchcodec.transforms.Resize(size=(height, width))]
        )
        decoder_resize_tv = VideoDecoder(
            video.path, transforms=[v2.Resize(size=(height, width))]
        )

        decoder_full = VideoDecoder(video.path)

        num_frames = len(decoder_resize)
        assert num_frames == len(decoder_full)

        for frame_index in [
            0,
            int(num_frames * 0.1),
            int(num_frames * 0.2),
            int(num_frames * 0.3),
            int(num_frames * 0.4),
            int(num_frames * 0.5),
            int(num_frames * 0.75),
            int(num_frames * 0.90),
            num_frames - 1,
        ]:
            frame_resize_tv = decoder_resize_tv[frame_index]
            frame_resize = decoder_resize[frame_index]
            assert_frames_equal(frame_resize_tv, frame_resize)

            frame_full = decoder_full[frame_index]

            frame_tv = v2.functional.resize(frame_full, size=(height, width))
            frame_tv_no_antialias = v2.functional.resize(
                frame_full, size=(height, width), antialias=False
            )

            expected_shape = (video.get_num_color_channels(), height, width)
            assert frame_resize.shape == expected_shape
            assert frame_tv.shape == expected_shape
            assert frame_tv_no_antialias.shape == expected_shape

            assert_tensor_close_on_at_least(
                frame_resize, frame_tv, percentage=99.8, atol=1
            )
            torch.testing.assert_close(frame_resize, frame_tv, rtol=0, atol=6)

            if height_scaling_factor < 1 or width_scaling_factor < 1:
                # Antialias only relevant when down-scaling!
                with pytest.raises(AssertionError, match="Expected at least"):
                    assert_tensor_close_on_at_least(
                        frame_resize, frame_tv_no_antialias, percentage=99, atol=1
                    )
                with pytest.raises(AssertionError, match="Tensor-likes are not close"):
                    torch.testing.assert_close(
                        frame_resize, frame_tv_no_antialias, rtol=0, atol=6
                    )

    def test_resize_fails(self):
        with pytest.raises(
            ValueError,
            match=r"must use bilinear interpolation",
        ):
            VideoDecoder(
                NASA_VIDEO.path,
                transforms=[
                    v2.Resize(
                        size=(100, 100), interpolation=v2.InterpolationMode.BICUBIC
                    )
                ],
            )

        with pytest.raises(
            ValueError,
            match=r"must have antialias enabled",
        ):
            VideoDecoder(
                NASA_VIDEO.path,
                transforms=[v2.Resize(size=(100, 100), antialias=False)],
            )

        with pytest.raises(
            ValueError,
            match=r"must have a size specified",
        ):
            VideoDecoder(
                NASA_VIDEO.path, transforms=[v2.Resize(size=None, max_size=100)]
            )

        with pytest.raises(
            ValueError,
            match=r"must have a \(height, width\) pair for the size",
        ):
            VideoDecoder(NASA_VIDEO.path, transforms=[v2.Resize(size=(100))])

        with pytest.raises(
            ValueError,
            match=r"must have a \(height, width\) pair for the size",
        ):
            VideoDecoder(
                NASA_VIDEO.path,
                transforms=[torchcodec.transforms.Resize(size=(100, 100, 100))],
            )

    def test_resize_non_32_aligned_input_width(self):
        assert NON_32_ALIGNED_WIDTH_VIDEO.get_width() % 32 != 0
        decoder = VideoDecoder(
            NON_32_ALIGNED_WIDTH_VIDEO.path,
            transforms=[torchcodec.transforms.Resize(size=(224, 224))],
        )
        assert decoder[0].shape == (3, 224, 224)

    @pytest.mark.parametrize(
        "height_scaling_factor, width_scaling_factor",
        ((0.5, 0.5), (0.25, 0.1), (1.0, 1.0), (0.15, 0.75)),
    )
    @pytest.mark.parametrize("video", [NASA_VIDEO, TEST_SRC_2_720P])
    def test_center_crop_torchvision(
        self,
        height_scaling_factor,
        width_scaling_factor,
        video,
    ):
        height = int(video.get_height() * height_scaling_factor)
        width = int(video.get_width() * width_scaling_factor)

        tc_center_crop = torchcodec.transforms.CenterCrop(size=(height, width))
        decoder_center_crop = VideoDecoder(video.path, transforms=[tc_center_crop])

        decoder_center_crop_tv = VideoDecoder(
            video.path,
            transforms=[v2.CenterCrop(size=(height, width))],
        )

        decoder_full = VideoDecoder(video.path)

        num_frames = len(decoder_center_crop_tv)
        assert num_frames == len(decoder_full)

        for frame_index in [
            0,
            int(num_frames * 0.25),
            int(num_frames * 0.5),
            int(num_frames * 0.75),
            num_frames - 1,
        ]:
            frame_center_crop = decoder_center_crop[frame_index]
            frame_center_crop_tv = decoder_center_crop_tv[frame_index]
            assert_frames_equal(frame_center_crop, frame_center_crop_tv)

            expected_shape = (video.get_num_color_channels(), height, width)
            assert frame_center_crop_tv.shape == expected_shape

            frame_full = decoder_full[frame_index]
            frame_tv = v2.CenterCrop(size=(height, width))(frame_full)
            assert_frames_equal(frame_center_crop, frame_tv)

    def test_center_crop_fails(self):
        with pytest.raises(
            ValueError,
            match=r"must have a \(height, width\) pair for the size",
        ):
            VideoDecoder(
                NASA_VIDEO.path,
                transforms=[torchcodec.transforms.CenterCrop(size=(100,))],
            )

    @pytest.mark.parametrize(
        "height_scaling_factor, width_scaling_factor",
        ((0.5, 0.5), (0.25, 0.1), (1.0, 1.0), (0.15, 0.75)),
    )
    @pytest.mark.parametrize("video", [NASA_VIDEO, TEST_SRC_2_720P])
    @pytest.mark.parametrize("seed", [0, 1234])
    def test_random_crop_torchvision(
        self,
        height_scaling_factor,
        width_scaling_factor,
        video,
        seed,
    ):
        height = int(video.get_height() * height_scaling_factor)
        width = int(video.get_width() * width_scaling_factor)

        # We want both kinds of RandomCrop objects to get arrive at the same
        # locations to crop, so we need to make sure they get the same random
        # seed. It's used in RandomCrop's _make_transform_spec() method, called
        # by the VideoDecoder.
        torch.manual_seed(seed)
        tc_random_crop = torchcodec.transforms.RandomCrop(size=(height, width))
        decoder_random_crop = VideoDecoder(video.path, transforms=[tc_random_crop])

        # Resetting manual seed for when TorchCodec's RandomCrop, created from
        # the TorchVision RandomCrop, is used inside of the VideoDecoder. It
        # needs to match the call above.
        torch.manual_seed(seed)
        decoder_random_crop_tv = VideoDecoder(
            video.path,
            transforms=[v2.RandomCrop(size=(height, width))],
        )

        decoder_full = VideoDecoder(video.path)

        num_frames = len(decoder_random_crop_tv)
        assert num_frames == len(decoder_full)

        for frame_index in [
            0,
            int(num_frames * 0.25),
            int(num_frames * 0.5),
            int(num_frames * 0.75),
            num_frames - 1,
        ]:
            frame_random_crop = decoder_random_crop[frame_index]
            frame_random_crop_tv = decoder_random_crop_tv[frame_index]
            assert_frames_equal(frame_random_crop, frame_random_crop_tv)

            expected_shape = (video.get_num_color_channels(), height, width)
            assert frame_random_crop_tv.shape == expected_shape

            # Resetting manual seed to make sure the invocation of the
            # TorchVision RandomCrop matches the two calls above.
            torch.manual_seed(seed)
            frame_full = decoder_full[frame_index]
            frame_tv = v2.RandomCrop(size=(height, width))(frame_full)
            assert_frames_equal(frame_random_crop, frame_tv)

    @pytest.mark.parametrize(
        "height_scaling_factor, width_scaling_factor",
        ((0.25, 0.1), (0.25, 0.25)),
    )
    def test_random_crop_nhwc(
        self,
        height_scaling_factor,
        width_scaling_factor,
    ):
        height = int(TEST_SRC_2_720P.get_height() * height_scaling_factor)
        width = int(TEST_SRC_2_720P.get_width() * width_scaling_factor)

        decoder = VideoDecoder(
            TEST_SRC_2_720P.path,
            transforms=[torchcodec.transforms.RandomCrop(size=(height, width))],
            dimension_order="NHWC",
        )

        num_frames = len(decoder)
        for frame_index in [
            0,
            int(num_frames * 0.25),
            int(num_frames * 0.5),
            int(num_frames * 0.75),
            num_frames - 1,
        ]:
            frame = decoder[frame_index]
            assert frame.shape == (height, width, 3)

    @pytest.mark.parametrize(
        "error_message, params",
        (
            ("must not specify padding", dict(size=(100, 100), padding=255)),
            (
                "must not specify pad_if_needed",
                dict(size=(100, 100), pad_if_needed=True),
            ),
            ("fill must be 0", dict(size=(100, 100), fill=255)),
            (
                "padding_mode must be constant",
                dict(size=(100, 100), padding_mode="edge"),
            ),
        ),
    )
    def test_random_crop_fails(self, error_message, params):
        with pytest.raises(
            ValueError,
            match=error_message,
        ):
            VideoDecoder(
                NASA_VIDEO.path,
                transforms=[v2.RandomCrop(**params)],
            )

    @pytest.mark.parametrize("seed", [0, 314])
    def test_random_crop_reusable_objects(self, seed):
        torch.manual_seed(seed)
        random_crop = torchcodec.transforms.RandomCrop(size=(99, 99))

        # Create a spec which causes us to calculate the random crop location.
        first_spec = random_crop._make_transform_spec((888, 888))

        # Create a spec again, which should calculate a different random crop
        # location. Despite having the same image size, the specs should be
        # different because the crop should be at a different location
        second_spec = random_crop._make_transform_spec((888, 888))
        assert first_spec != second_spec

        # Create a spec again, but with a different image size. The specs should
        # obviously be different, but the original image size should not be in
        # the spec at all.
        third_spec = random_crop._make_transform_spec((777, 777))
        assert third_spec != first_spec
        assert "888" not in third_spec

    @pytest.mark.parametrize(
        "resize, random_crop",
        [
            (torchcodec.transforms.Resize, torchcodec.transforms.RandomCrop),
            (v2.Resize, v2.RandomCrop),
        ],
    )
    def test_transform_pipeline(self, resize, random_crop):
        decoder = VideoDecoder(
            TEST_SRC_2_720P.path,
            transforms=[
                # resized to bigger than original
                resize(size=(2160, 3840)),
                # crop to smaller than the resize, but still bigger than original
                random_crop(size=(1080, 1920)),
            ],
        )

        num_frames = len(decoder)
        for frame_index in [
            0,
            int(num_frames * 0.25),
            int(num_frames * 0.5),
            int(num_frames * 0.75),
            num_frames - 1,
        ]:
            frame = decoder[frame_index]
            assert frame.shape == (TEST_SRC_2_720P.get_num_color_channels(), 1080, 1920)

    def test_transform_fails(self):
        with pytest.raises(
            ValueError,
            match="Unsupported transform",
        ):
            VideoDecoder(NASA_VIDEO.path, transforms=[v2.RandomHorizontalFlip(p=1.0)])

    @pytest.mark.parametrize("output_dtype", (torch.uint8, torch.float32))
    # An output width that is a multiple of 32 lets the decoder use swscale for
    # NASA_VIDEO and NASA_VIDEO_HDR. Other widths use filtergraph.
    @pytest.mark.parametrize("aligned_width", (True, False))
    # NASA_VIDEO_HDR is 10-bit. NASA_VIDEO_ROTATED has rotation metadata, so
    # the decoder rotates the frame before the resize.
    @pytest.mark.parametrize("video", (NASA_VIDEO, NASA_VIDEO_HDR, NASA_VIDEO_ROTATED))
    def test_resize_before_color_conversion(self, video, aligned_width, output_dtype):
        height = video.get_height() // 2
        width = video.get_width() // 2 // 32 * 32 + (0 if aligned_width else 8)
        decoder = VideoDecoder(
            video.path,
            transforms=[torchcodec.transforms.Resize(size=(height, width))],
            output_dtype=output_dtype,
            resize_before_color_conversion=True,
        )
        decoder_full = VideoDecoder(video.path, output_dtype=output_dtype)

        for frame_index in (0, len(decoder) // 2, len(decoder) - 1):
            frame = decoder[frame_index]
            assert frame.shape == (3, height, width)
            assert frame.dtype == output_dtype

            # Resizing in the decoded pixel format, then converting to RGB, is
            # close to resizing the RGB frame, but not the same.
            frame_tv = v2.functional.resize(
                decoder_full[frame_index], size=(height, width)
            )
            atol = 3 if output_dtype == torch.uint8 else 3 / 255
            assert_tensor_close_on_at_least(frame, frame_tv, percentage=85, atol=atol)

    def test_resize_before_color_conversion_nhwc_and_batch_apis(self):
        def make_decoder(dimension_order):
            return VideoDecoder(
                NASA_VIDEO.path,
                transforms=[torchcodec.transforms.Resize(size=(135, 240))],
                dimension_order=dimension_order,
                resize_before_color_conversion=True,
            )

        nchw, nhwc = make_decoder("NCHW"), make_decoder("NHWC")
        indices = [0, 5, 10]
        expected = torch.stack([nchw[i] for i in indices])
        assert expected.shape == (3, 3, 135, 240)
        assert_frames_equal(nchw.get_frames_at(indices).data, expected)
        assert_frames_equal(nchw[0:11:5], expected)
        assert_frames_equal(
            nhwc.get_frames_at(indices).data, expected.permute(0, 2, 3, 1)
        )

    def test_resize_before_color_conversion_with_other_transforms(self):
        # The resize happens before the color conversion; the following
        # transforms are applied to the RGB frame, as usual.
        decoder = VideoDecoder(
            NASA_VIDEO.path,
            transforms=[v2.Resize(size=(135, 240)), v2.CenterCrop(size=(100, 100))],
            resize_before_color_conversion=True,
        )
        resized = VideoDecoder(
            NASA_VIDEO.path,
            transforms=[torchcodec.transforms.Resize(size=(135, 240))],
            resize_before_color_conversion=True,
        )
        assert_frames_equal(
            decoder[0], v2.functional.center_crop(resized[0], output_size=(100, 100))
        )

    @pytest.mark.parametrize(
        "transforms",
        (
            None,
            [],
            [torchcodec.transforms.CenterCrop(size=(100, 100))],
            [
                torchcodec.transforms.CenterCrop(size=(100, 100)),
                torchcodec.transforms.Resize(size=(50, 50)),
            ],
            [v2.CenterCrop(size=(100, 100))],
        ),
    )
    def test_resize_before_color_conversion_fails(self, transforms):
        with pytest.raises(ValueError, match="requires the first of the transforms"):
            VideoDecoder(
                NASA_VIDEO.path,
                transforms=transforms,
                resize_before_color_conversion=True,
            )

    @pytest.mark.parametrize("value", (1, "True", None))
    def test_resize_before_color_conversion_not_a_bool(self, value):
        with pytest.raises(TypeError, match="must be a bool"):
            VideoDecoder(
                NASA_VIDEO.path,
                transforms=[torchcodec.transforms.Resize(size=(100, 100))],
                resize_before_color_conversion=value,
            )


class TestCoreVideoDecoderTransformOps:
    def get_num_frames_core_ops(self, video):
        decoder = create_from_file(str(video.path))
        add_video_stream(decoder)
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        num_frames = metadata_dict["numFramesFromHeader"]
        assert num_frames is not None
        return num_frames

    @pytest.mark.parametrize("video", [NASA_VIDEO, H265_VIDEO, AV1_VIDEO])
    def test_color_conversion_library(self, video):
        num_frames = self.get_num_frames_core_ops(video)

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
        add_video_stream(
            auto_decoder,
        )

        filtergraph_frame0, *_ = get_frame_at_index(filtergraph_decoder, frame_index=0)
        auto_frame0, *_ = get_frame_at_index(auto_decoder, frame_index=0)
        assert_frames_equal(filtergraph_frame0, auto_frame0)

    def _resized_frame0(self, video, transform_specs, color_conversion_library):
        decoder = create_from_file(str(video.path))
        _add_video_stream(
            decoder,
            transform_specs=transform_specs,
            color_conversion_library=color_conversion_library,
        )
        frame, *_ = get_frame_at_index(decoder, frame_index=0)
        return frame

    # 96 is a multiple of 32, 100 is not. The core uses the requested library
    # either way.
    @pytest.mark.parametrize("width", (96, 100))
    @pytest.mark.parametrize("video", (NASA_VIDEO, H265_10BITS))
    def test_resize_before_color_conversion_library_parity(self, video, width):
        spec = f"resize_before_color_conversion, 100, {width}"
        filtergraph, swscale = (
            self._resized_frame0(video, spec, library)
            for library in ("filtergraph", "swscale")
        )
        assert filtergraph.shape == (3, 100, width)
        # Bit-exact on FFmpeg 7.1. The tolerance is for other FFmpeg versions,
        # where filtergraph may go through a different swscale entry point.
        assert_tensor_close_on_at_least(filtergraph, swscale, percentage=99, atol=1)

    def test_resize_before_color_conversion_must_be_first(self):
        with pytest.raises(RuntimeError, match="must be the first transform"):
            self._resized_frame0(
                NASA_VIDEO,
                "crop, 100, 100, 0, 0;resize_before_color_conversion, 50, 50",
                None,
            )

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

            if torchcodec.ffmpeg_major_version <= 4 and get_ffmpeg_minor_version() <= 1:
                # FFmpeg version 4.1 and before appear to have a different
                # resize implementation.
                torch.testing.assert_close(frame_resize, frame_ref, rtol=0, atol=2)
            else:
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
            match="must have 3 elements",
        ):
            add_video_stream(
                decoder, transform_specs="resize_before_color_conversion, 100"
            )

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
            match="x start position, 9999, out of bounds",
        ):
            decoder = create_from_file(str(NASA_VIDEO.path))
            add_video_stream(decoder, transform_specs="crop, 100, 100, 9999, 100")

        with pytest.raises(
            RuntimeError,
            match=r"Crop output height \(999\) is greater than input height \(270\)",
        ):
            decoder = create_from_file(str(NASA_VIDEO.path))
            add_video_stream(decoder, transform_specs="crop, 999, 100, 100, 100")
