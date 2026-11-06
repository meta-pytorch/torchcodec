# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
===============================================
Decoding packets that don't come from a Demuxer
===============================================

.. currentmodule:: torchcodec.decoders

.. important::

   **The low-level APIs are in beta.** Their signatures and semantics may still
   change slightly, in response to user feedback. Please `share your feedback
   <https://github.com/meta-pytorch/torchcodec/issues?q=is:open+is:issue>`__!

Sometimes, you don't have a file to demux. Maybe your packets were demuxed
ahead of time and stored in a database, maybe they arrive over the network, or
maybe they come from a container format that TorchCodec can't read. You can
still decode them: a :class:`VideoPacketDecoder` doesn't need a
:class:`Demuxer`. All it needs are the :class:`VideoCodecParameters` of the
stream, and :class:`Packet` objects, which you can build from the compressed
data yourself.

In this tutorial, we'll decode packets that an earlier job demuxed and stored,
without the original file.
"""

# %%
# Some boilerplate first: a test video, and the device we'll run on.
import subprocess
import tempfile
from pathlib import Path

import torch

# sphinx_gallery_thumbnail_path = '_static/thumbnails/grumps_low_level_demuxer_free.jpg'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device = }")

temp_dir = Path(tempfile.mkdtemp())
video_path = temp_dir / "video.mp4"
subprocess.run(
    [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", "-bf", "2",
        str(video_path),
    ],
    check=True,
)

# %%
# Storing the packets
# -------------------
#
# Let's play the earlier job: it demuxes the video and stores, for each packet,
# its data, its :term:`pts` and its flags, which is the data we'll later need to
# reconstruct such Packets. It also stores the values a decoder needs about the
# stream, which we'll look at below.
from torchcodec.decoders import Demuxer

demuxer = Demuxer(video_path)
stream = demuxer.streams[0]
rows = [(bytes(packet.data.numpy()), packet.pts, packet.flags) for packet in demuxer]
stream_info = {
    "codec": stream.codec_parameters.codec,
    "time_base": stream.codec_parameters.time_base,
    "extradata": stream.codec_parameters.extradata,
    "width": stream.codec_parameters.width,
    "height": stream.codec_parameters.height,
    "pixel_format": stream.codec_parameters.pixel_format,
    "r_frame_rate": stream.codec_parameters.r_frame_rate,
}
print(f"{len(rows)} packets stored")

# %%
# From here on, there's no :class:`Demuxer` and no file: only ``rows`` and
# ``stream_info``, which play the role of our database.
del demuxer, stream

# %%
# Codec parameters
# ----------------
#
# A decoder needs to know about the stream it decodes: its codec, the unit of
# the packets' timestamps, and some codec-specific values. That's what
# :class:`VideoCodecParameters` holds. Here, we build them from the stored
# values:
from torchcodec.decoders import VideoCodecParameters

codec_parameters = VideoCodecParameters(
    stream_info["codec"],
    time_base=stream_info["time_base"],
    extradata=stream_info["extradata"],
    width=stream_info["width"],
    height=stream_info["height"],
    pixel_format=stream_info["pixel_format"],
    r_frame_rate=stream_info["r_frame_rate"],
)
print(codec_parameters)

# %%
# Which fields you need depends on the codec. For H.264 in MP4, the decoder
# needs the ``extradata``: it's where MP4 stores the codec's setup data,
# outside of the packets. To decode on CUDA, ``width``, ``height`` and
# ``pixel_format`` are needed too, and ``r_frame_rate`` gives the frames their
# durations. To find out what to set for your own packets, print the codec
# parameters a :class:`Demuxer` reports for a similar file:
# ``print(Demuxer(path).streams[0].codec_parameters)``.
#
# .. note::
#
#    If the packets are demuxed with TorchCodec in the first place, as above,
#    it's simpler and safer to store the whole :class:`VideoCodecParameters`
#    object, e.g. pickled, rather than a few of its fields: it holds everything
#    the decoder may need, including fields you wouldn't think of.

# %%
# Decoding the packets
# --------------------
#
# :meth:`Packet.from_bytes` builds a packet from its data, and
# :meth:`Packet.from_tensor` does the same from a 1D uint8 tensor. Only the data
# is required, but the pts is what gives the decoded frames their timestamps.
# **Unlike in most TorchCodec APIs, timestamps are integers**, in units of the
# stream's time base. The rest is the usual decoding loop:
from torchcodec.decoders import ColorConverter, Packet, VideoPacketDecoder

packets = [Packet.from_bytes(data, pts=pts, flags=flags) for data, pts, flags in rows]

packet_decoder = VideoPacketDecoder(codec_parameters, device=device)
color_converter = ColorConverter(device=device)

frames = []
for packet in packets:
    frames += [color_converter.convert(f) for f in packet_decoder.decode(packet)]
frames += [color_converter.convert(f) for f in packet_decoder.drain()]
print(f"{len(frames)} frames, {frames[0].data.shape = }, {frames[0].data.device = }")

# %%
# Matching frames to packets
# --------------------------
#
# The frames don't come out of the decoder in the order the packets went in.
# But :attr:`RawFrame.pts` is the pts of the packet the frame was decoded from,
# and both are integers: a frame can be matched exactly to its packet, e.g. to
# look up what you stored along with it.

# %%
# The codec parameters and the packets can also be pickled, e.g. to demux in
# one process and decode in another: see the last section of
# :ref:`sphx_glr_generated_examples_low_level_pipelines.py`.
