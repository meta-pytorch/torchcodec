# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
=======================================
Encoding video frames with VideoEncoder
=======================================

In this example, we'll learn how to encode video frames to a file or to raw
bytes using the :class:`~torchcodec.encoders.VideoEncoder` class.
"""

# %%
# First, we'll download a video and decode some frames to tensors.
# These will be the input to the :class:`~torchcodec.encoders.VideoEncoder`. For more details on decoding,
# see :ref:`sphx_glr_generated_examples_decoding_basic_example.py`.
# Otherwise, skip ahead to :ref:`creating_encoder`.

import requests
from torchcodec.decoders import VideoDecoder
from IPython.display import Video


def play_video(encoded_bytes):
    return Video(
        data=encoded_bytes.numpy().tobytes(),
        embed=True,
        width=640,
        height=360,
        mimetype="video/mp4",
    )


# Video source: https://www.pexels.com/video/adorable-cats-on-the-lawn-4977395/
# Author: Altaf Shah.
url = "https://videos.pexels.com/video-files/4977395/4977395-hd_1920_1080_24fps.mp4"

response = requests.get(url, headers={"User-Agent": ""})
if response.status_code != 200:
    raise RuntimeError(f"Failed to download video. {response.status_code = }.")

raw_video_bytes = response.content

decoder = VideoDecoder(raw_video_bytes)
frames = decoder.get_frames_in_range(0, 60).data  # Get first 60 frames
frame_rate = decoder.metadata.average_fps

# %%
# .. _creating_encoder:
#
# Creating an encoder
# -------------------
#
# Let's instantiate a :class:`~torchcodec.encoders.VideoEncoder`. We will need to provide
# the frames to be encoded as a 4D tensor of shape
# ``(num_frames, num_channels, height, width)`` with values in the ``[0, 255]``
# range and ``torch.uint8`` dtype. We will also need to provide the frame rate of the input
# video.
#
# .. note::
#
#     The ``frame_rate`` parameter corresponds to the frame rate of the
#     *input* video. It will also be used for the frame rate of the *output* encoded video.
from torchcodec.encoders import VideoEncoder

print(f"{frames.shape = }, {frames.dtype = }")
print(f"{frame_rate = } fps")

encoder = VideoEncoder(frames=frames, frame_rate=frame_rate)

# %%
# Encoding to file, bytes, or file-like
# -------------------------------------
#
# :class:`~torchcodec.encoders.VideoEncoder` supports encoding frames into a
# file via the :meth:`~torchcodec.encoders.VideoEncoder.to_file` method, to
# file-like objects via the :meth:`~torchcodec.encoders.VideoEncoder.to_file_like`
# method, or to raw bytes via :meth:`~torchcodec.encoders.VideoEncoder.to_tensor`.
# For now we will use :meth:`~torchcodec.encoders.VideoEncoder.to_tensor`, so we
# can easily inspect and display the encoded video.

encoded_frames = encoder.to_tensor(format="mp4")
play_video(encoded_frames)

# %%
#
# Now that we have encoded data, we can decode it back to verify the
# round-trip encode/decode process works as expected:

decoder_verify = VideoDecoder(encoded_frames)
decoded_frames = decoder_verify.get_frames_in_range(0, 60).data

print(f"Re-decoded video: {decoded_frames.shape = }")
print(f"Original frames: {frames.shape = }")

# %%
# .. _codec_selection:
#
# Codec Selection
# ---------------
#
# By default, the codec used is selected automatically using the file extension provided
# in the ``dest`` parameter for the :meth:`~torchcodec.encoders.VideoEncoder.to_file` method,
# or using the ``format`` parameter for the
# :meth:`~torchcodec.encoders.VideoEncoder.to_file_like` and
# :meth:`~torchcodec.encoders.VideoEncoder.to_tensor` methods.
#
# For example, when encoding to MP4 format, the default codec is typically ``H.264``.
#
# To use a codec other than the default, use the ``codec`` parameter.
# You can specify either a specific codec implementation (e.g., ``"libx264"``)
# or a codec specification (e.g., ``"h264"``). Different codecs offer
# different tradeoffs between quality, file size, and encoding speed.
#
# .. note::
#
#     To see available encoders on your system, run ``ffmpeg -encoders``.
#
# Let's encode the same frames using different codecs:

import tempfile
from pathlib import Path

# H.264 encoding
h264_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder.to_file(h264_output, codec="libx264")

# H.265 encoding
hevc_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder.to_file(hevc_output, codec="hevc")

# Now let's use ffprobe to verify the codec used in the output files
import subprocess

for output, name in [(h264_output, "h264_output"), (hevc_output, "hevc_output")]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            output,
        ],
        capture_output=True,
        text=True,
    )
    print(f"Codec used in {name}: {result.stdout.strip()}")


# %%
# .. _pixel_format:
#
# Pixel Format
# ------------
#
# The ``pixel_format`` parameter controls the color sampling (chroma subsampling)
# of the output video. This affects both quality and file size.
#
# Common pixel formats:
#
# - ``"yuv420p"`` - 4:2:0 chroma subsampling (standard quality, smaller file size, widely compatible)
# - ``"yuv444p"`` - 4:4:4 chroma subsampling (full chroma resolution, higher quality, larger file size)
#
# Most playback devices and platforms support ``yuv420p``, making it the most
# common choice for video encoding.
#
# .. note::
#
#     Pixel format support depends on the codec used. Use ``ffmpeg -h encoder=<codec_name>``
#     to check available options for your selected codec.

# Standard pixel format
yuv420_encoded_frames = encoder.to_tensor(
    format="mp4", codec="libx264", pixel_format="yuv420p"
)
play_video(yuv420_encoded_frames)

# %%
# .. _crf:
#
# CRF (Constant Rate Factor)
# --------------------------
#
# The ``crf`` parameter controls video quality, where lower values produce higher quality output.
#
# For example, with the commonly used H.264 codec, ``libx264``:
#
# - Values range from 0 (lossless) to 51 (worst quality)
# - Values 17 or 18 are considered visually lossless, and the default is 23.
#
# .. note::
#
#     The range and interpretation of CRF values depend on the codec used, and
#     not all codecs support CRF. Use ``ffmpeg -h encoder=<codec_name>`` to
#     check available options for your selected codec.
#

# High quality (low CRF)
high_quality_output = encoder.to_tensor(format="mp4", codec="libx264", crf=0)
play_video(high_quality_output)

# %%
# Low quality (high CRF)
low_quality_output = encoder.to_tensor(format="mp4", codec="libx264", crf=50)
play_video(low_quality_output)


# %%
# .. _preset:
#
# Preset
# ------
#
# The ``preset`` parameter controls the tradeoff between encoding speed and file compression.
# Faster presets encode faster but produce larger files, while slower
# presets take more time to encode but result in better compression.
#
# For example, with the commonly used H.264 codec, ``libx264`` presets include
# ``"ultrafast"`` (fastest), ``"fast"``, ``"medium"`` (default), ``"slow"``, and
# ``"veryslow"`` (slowest, best compression). See the
# `H.264 Video Encoding Guide <https://trac.ffmpeg.org/wiki/Encode/H.264#a2.Chooseapresetandtune>`_
# for additional details.
#
# .. note::
#
#     Not all codecs support the ``presets`` option. Use ``ffmpeg -h encoder=<codec_name>``
#     to check available options for your selected codec.
#

# Fast encoding with a larger file size
fast_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder.to_file(fast_output, codec="libx264", preset="ultrafast")
print(f"Size of fast encoded file: {Path(fast_output).stat().st_size} bytes")

# Slow encoding for a smaller file size
slow_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder.to_file(slow_output, codec="libx264", preset="veryslow")
print(f"Size of slow encoded file: {Path(slow_output).stat().st_size} bytes")

# %%
# .. _extra_options:
#
# Extra Options
# -------------
#
# The ``extra_options`` parameter accepts a dictionary of codec-specific options
# that would normally be set via FFmpeg command-line arguments. This enables
# control of encoding settings beyond the common parameters.
#
# For example, some potential extra options for the commonly used H.264 codec, ``libx264`` include:
#
# - ``"g"`` - GOP (Group of Pictures) size / keyframe interval
# - ``"max_b_frames"`` - Maximum number of B-frames between I and P frames
# - ``"tune"`` - Tuning preset (e.g., ``"film"``, ``"animation"``, ``"grain"``)
#
# .. note::
#
#     Use ``ffmpeg -h encoder=<codec_name>`` to see all available options for
#     a specific codec.
#


custom_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder.to_file(
    custom_output,
    codec="libx264",
    extra_options={
        "g": 50,               # Keyframe every 50 frames
        "max_b_frames": 0,     # Disable B-frames for faster decoding
        "tune": "fastdecode",  # Optimize for fast decoding
    },
)

# %%
