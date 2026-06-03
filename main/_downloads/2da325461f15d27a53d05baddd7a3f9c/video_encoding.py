# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
===============================
Encoding video with the Encoder
===============================

In this example, we'll learn how to encode video frames using the
:class:`~torchcodec.encoders.Encoder` class, and we'll explore the different
encoding parameters available.

.. tip::

   :class:`~torchcodec.encoders.VideoEncoder` and
   :class:`~torchcodec.encoders.AudioEncoder` are simpler, single-stream
   encoders. They are convenient for simple one-shot encoding, but the
   :class:`~torchcodec.encoders.Encoder` class is more flexible and supports
   the same encoding options. See
   :ref:`sphx_glr_generated_examples_encoding_multi_stream_encoding.py` for a
   multi-stream encoding tutorial.
"""

# %%
# First, let's download a video and decode some frames to tensors.
# These will be the input for our encoder. For more details on decoding,
# see :ref:`sphx_glr_generated_examples_decoding_basic_example.py`.
# Otherwise, skip ahead to :ref:`creating_encoder`.

import io

import requests
from torchcodec.decoders import VideoDecoder
from IPython.display import Video

# sphinx_gallery_thumbnail_path = '_static/thumbnails/not_grumps_encoding_video.jpg'


def play_video(encoded_bytes):
    return Video(
        data=encoded_bytes,
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
# Creating an encoder and encoding to a file
# -------------------------------------------
#
# Let's instantiate an :class:`~torchcodec.encoders.Encoder`, add a video
# stream, and encode to a file. We use the
# :meth:`~torchcodec.encoders.Encoder.open_file` method as a context manager to
# ensure the output is properly flushed and closed.
#
# .. note::
#
#     The ``frame_rate`` parameter corresponds to the frame rate of the
#     *input* video. It will also be used for the frame rate of the *output*
#     encoded video.

import tempfile
from pathlib import Path
from torchcodec.encoders import Encoder

print(f"{frames.shape = }, {frames.dtype = }")
print(f"{frame_rate = } fps")

output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder = Encoder()
video_stream = encoder.add_video(
    height=frames.shape[2], width=frames.shape[3], frame_rate=frame_rate
)
with encoder.open_file(output_path):
    video_stream.add_frames(frames)
    # More frames can be submitted by calling video_stream.add_frames

print(f"Encoded to {output_path}, size: {Path(output_path).stat().st_size} bytes")

# %%
# Now that we have encoded data, let's decode it back to verify the
# round-trip encode/decode process works as expected:

decoder_verify = VideoDecoder(output_path)
decoded_frames = decoder_verify.get_frames_in_range(0, 60).data

print(f"Re-decoded video: {decoded_frames.shape = }")
print(f"Original frames: {frames.shape = }")

# %%
# .. _cuda_encoding:
#
# CUDA Encoding
# -------------
#
# To encode on GPU, pass ``device="cuda"`` to
# :meth:`~torchcodec.encoders.Encoder.add_video`, and feed CUDA tensors to
# :meth:`~torchcodec.encoders.VideoStream.add_frames`. This can result in
# significantly faster encoding than CPU. The encoder will automatically select
# a CUDA-compatible codec like ``h264_nvenc`` or ``hevc_nvenc``.
#
# .. note::
#
#     On GPU, the pixel format is always set to ``nv12`` (which does equivalent
#     chroma subsampling to ``yuv420p``). The ``pixel_format`` parameter is not
#     supported for GPU encoding.
#
# .. code-block:: python
#
#     gpu_frames = frames.to("cuda")  # Move frames to GPU
#     encoder = Encoder()
#     video_stream = encoder.add_video(
#         height=gpu_frames.shape[2], width=gpu_frames.shape[3],
#         frame_rate=frame_rate, device="cuda",
#     )
#     with encoder.open_file("output.mp4"):
#         video_stream.add_frames(gpu_frames)
#
# That's it! The rest of the encoding process is the same as on CPU.

# %%
# .. _codec_selection:
#
# Codec Selection
# ---------------
#
# By default, the codec is selected automatically based on the container format.
# For example, when encoding to MP4 format, the default codec is typically
# ``H.264``.
#
# To use a codec other than the default, use the ``codec`` parameter in
# :meth:`~torchcodec.encoders.Encoder.add_video`. You can specify either a
# specific codec implementation (e.g., ``"libx264"``) or a codec specification
# (e.g., ``"h264"``). Different codecs offer different tradeoffs between
# quality, file size, and encoding speed.
#
# .. note::
#
#     To see available encoders on your system, run ``ffmpeg -encoders``.
#
# Let's encode the same frames using different codecs:

# H.264 encoding
h264_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder = Encoder()
vs = encoder.add_video(
    height=frames.shape[2], width=frames.shape[3],
    frame_rate=frame_rate, codec="libx264",
)
with encoder.open_file(h264_output):
    vs.add_frames(frames)

# H.265 encoding
hevc_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder = Encoder()
vs = encoder.add_video(
    height=frames.shape[2], width=frames.shape[3],
    frame_rate=frame_rate, codec="hevc",
)
with encoder.open_file(hevc_output):
    vs.add_frames(frames)

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
buf = io.BytesIO()
encoder = Encoder()
vs = encoder.add_video(
    height=frames.shape[2], width=frames.shape[3],
    frame_rate=frame_rate, codec="libx264", pixel_format="yuv420p",
)
with encoder.open_file_like(buf, format="mp4"):
    vs.add_frames(frames)

play_video(buf.getvalue())

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
buf = io.BytesIO()
encoder = Encoder()
vs = encoder.add_video(
    height=frames.shape[2], width=frames.shape[3],
    frame_rate=frame_rate, codec="libx264", crf=0,
)
with encoder.open_file_like(buf, format="mp4"):
    vs.add_frames(frames)

# play_video is disabled because crf=0 creates a 50+ Mb video that we don't want
# to check into our docs
# play_video(buf.getvalue())

# %%

# Low quality (high CRF)
buf = io.BytesIO()
encoder = Encoder()
vs = encoder.add_video(
    height=frames.shape[2], width=frames.shape[3],
    frame_rate=frame_rate, codec="libx264", crf=50,
)
with encoder.open_file_like(buf, format="mp4"):
    vs.add_frames(frames)

play_video(buf.getvalue())


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
encoder = Encoder()
vs = encoder.add_video(
    height=frames.shape[2], width=frames.shape[3],
    frame_rate=frame_rate, codec="libx264", preset="ultrafast",
)
with encoder.open_file(fast_output):
    vs.add_frames(frames)

print(f"Size of fast encoded file: {Path(fast_output).stat().st_size} bytes")

# Slow encoding for a smaller file size
slow_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder = Encoder()
vs = encoder.add_video(
    height=frames.shape[2], width=frames.shape[3],
    frame_rate=frame_rate, codec="libx264", preset="veryslow",
)
with encoder.open_file(slow_output):
    vs.add_frames(frames)

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
encoder = Encoder()
vs = encoder.add_video(
    height=frames.shape[2], width=frames.shape[3],
    frame_rate=frame_rate, codec="libx264",
    extra_options={
        "g": 50,                # Keyframe every 50 frames
        "max_b_frames": 0,      # Disable B-frames for faster decoding
        "tune": "fastdecode",   # Optimize for fast decoding
    },
)
with encoder.open_file(custom_output):
    vs.add_frames(frames)

# %%
