# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
===================
Decoding HDR videos
===================

In this example, we'll learn how to decode HDR (High Dynamic Range) videos
using the ``output_dtype`` parameter of the
:class:`~torchcodec.decoders.VideoDecoder` class.

.. note::

   The ``output_dtype`` parameter is in beta. Its behavior may change in future
   versions.

HDR videos typically encode pixel data with more than 8 bits per channel (e.g.
10 or 12 bits). This allows them to represent a wider range of colors and
brightness levels. When decoding such content, it is generally desirable to
preserve that extra precision by decoding into ``float32`` tensors rather than
the default ``uint8``.
"""

# %%
# Generating test videos with FFmpeg
# -----------------------------------
#
# First, we'll use FFmpeg to create two short synthetic videos: an SDR video
# (standard 8-bit H.264) and an HDR video (10-bit H.265 with BT.2020 color
# primaries and SMPTE ST 2084 / PQ transfer characteristics, which is a common
# HDR format).

import subprocess
import tempfile
from pathlib import Path

import torch

temp_dir = tempfile.mkdtemp()
sdr_video_path = Path(temp_dir) / "sdr_video.mp4"
hdr_video_path = Path(temp_dir) / "hdr_video.mp4"

# Generate a short SDR video (standard 8-bit H.264)
subprocess.run(
    [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=duration=2:size=320x180:rate=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        str(sdr_video_path),
    ],
    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)

# Generate a short HDR video (10-bit H.265 with BT.2020 + PQ)
subprocess.run(
    [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=duration=2:size=320x180:rate=30",
        "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
        "-x265-params",
        "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited",
        "-preset", "fast", "-crf", "23",
        str(hdr_video_path),
    ],
    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)

# %%
# Default behavior: ``uint8`` output
# -----------------------------------
#
# By default, the decoder outputs frames as ``torch.uint8`` tensors with
# values in [0, 255]. This works well for SDR content, but for HDR videos
# the 10-bit (or 12-bit) pixel values get quantized down to 8 bits, losing
# precision:

from torchcodec.decoders import VideoDecoder

sdr_decoder = VideoDecoder(sdr_video_path)
sdr_frame = sdr_decoder[0]

print(f"SDR pixel format: {sdr_decoder.metadata.pixel_format}")
print(f"SDR frame dtype: {sdr_frame.dtype}")
print(f"SDR frame value range: [{sdr_frame.min()}, {sdr_frame.max()}]")

# %%
hdr_decoder = VideoDecoder(hdr_video_path)
hdr_frame = hdr_decoder[0]

print(f"HDR pixel format: {hdr_decoder.metadata.pixel_format}")
print(f"HDR frame dtype: {hdr_frame.dtype}")
print(f"HDR frame value range: [{hdr_frame.min()}, {hdr_frame.max()}]")

# %%
# Both SDR and HDR videos produce ``uint8`` frames. For the HDR video, this
# means precision is lost: the original 10-bit values (0-1023) are squeezed
# into 8-bit (0-255).

# %%
# Using ``output_dtype=torch.float32``
# -------------------------------------
#
# To preserve the full precision of HDR content, set
# ``output_dtype=torch.float32``. This produces frames with values in [0, 1].
# This can also be used on SDR content if you want normalized float values:

sdr_decoder_float = VideoDecoder(sdr_video_path, output_dtype=torch.float32)
sdr_frame_float = sdr_decoder_float[0]

print(f"SDR frame as float32: dtype={sdr_frame_float.dtype}, "
      f"range=[{sdr_frame_float.min():.4f}, {sdr_frame_float.max():.4f}]")

# %%
hdr_decoder_float = VideoDecoder(hdr_video_path, output_dtype=torch.float32)
hdr_frame_float = hdr_decoder_float[0]

print(f"HDR frame as float32: dtype={hdr_frame_float.dtype}, "
      f"range=[{hdr_frame_float.min():.4f}, {hdr_frame_float.max():.4f}]")

# %%
# Using ``output_dtype="auto"``
# -----------------------------
#
# When working with a mix of SDR and HDR videos, you can use
# ``output_dtype="auto"`` to let the decoder choose the output dtype
# automatically. SDR content will be decoded as ``uint8``, and HDR content
# (i.e. videos with more than 8 bits per channel) will be decoded as
# ``float32``:

auto_sdr_decoder = VideoDecoder(sdr_video_path, output_dtype="auto")
auto_hdr_decoder = VideoDecoder(hdr_video_path, output_dtype="auto")

print(f"SDR video with 'auto': {auto_sdr_decoder[0].dtype}")
print(f"HDR video with 'auto': {auto_hdr_decoder[0].dtype}")

# %%
# Inspecting HDR metadata
# -----------------------
#
# You can inspect color-related metadata to understand the HDR characteristics
# of a video. Key fields are ``pixel_format``, ``color_primaries``,
# ``color_space``, and ``color_transfer_characteristic``:

print(f"Pixel format: {auto_hdr_decoder.metadata.pixel_format}")
print(f"Color primaries: {auto_hdr_decoder.metadata.color_primaries}")
print(f"Color space: {auto_hdr_decoder.metadata.color_space}")
print(f"Transfer characteristic: {auto_hdr_decoder.metadata.color_transfer_characteristic}")

# %%
# We can verify that these match the raw stream properties reported by
# ``ffprobe``:

result = subprocess.run(
    [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=pix_fmt,color_primaries,color_space,color_transfer",
        "-of", "default=noprint_wrappers=1",
        str(hdr_video_path),
    ],
    capture_output=True, text=True, check=True,
)
print(result.stdout)

# %%
import shutil
shutil.rmtree(temp_dir)

# sphinx_gallery_thumbnail_path = '_static/thumbnails/grumps_hdr.jpg'
