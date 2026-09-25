# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
================================
Raw frames and raw audio samples
================================

.. currentmodule:: torchcodec.decoders._blocks

.. warning::

   **The Blocks APIs are under active construction.** They are private
   and unreleased. Signatures and semantics may change without notice. This
   tutorial only exists to show what they will eventually make possible.

In this tutorial, we'll skip the conversion stage of a blocks pipeline and read
the decoder's own YUV planes and audio samples directly, at the source's own
precision.

The last stage of a blocks pipeline - a :class:`ColorConverter` for video, an
:class:`AudioConverter` for audio - is optional. If you stop before it, you get
what the decoder actually produced: YUV planes in the codec's own pixel format,
and audio samples in the codec's own sample type, with no conversion, no
normalisation, and no copy.

This is worth doing when the conversion you want isn't the one the converters
perform. You may want to apply a custom colorspace, train a model directly in
YUV space, write a kernel fused with the first layer of your model, decode a
10-bit HDR source without flattening it to `uint8` or `float32`, or scale
integer audio samples yourself.

This tutorial assumes you are familiar with the three stages described in
:ref:`sphx_glr_generated_examples_blocks_basics.py`.
"""

# %%
# Boilerplate: a test video, and the device we'll run on.
import subprocess
import tempfile
from pathlib import Path

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device = }")

temp_dir = Path(tempfile.mkdtemp())
video_path = temp_dir / "video.mp4"
subprocess.run(
    [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        str(video_path),
    ],
    check=True,
)


def decode_raw(demuxer, packet_decoder):
    for packet in demuxer:
        yield from packet_decoder.decode(packet)
    yield from packet_decoder.drain()


# %%
# Raw frames
# ----------
#
# A :class:`VideoPacketDecoder` produces :class:`RawFrame` objects, and a
# :class:`RawFrame` can hand out the decoder's own planes as tensor views, with
# no copy and no conversion. It also carries everything you need to interpret
# them.
from torchcodec.decoders._blocks import ColorConverter, Demuxer

demuxer = Demuxer(video_path)
packet_decoder = demuxer.streams[0].make_decoder(device=device)
raw_frame = next(decode_raw(demuxer, packet_decoder))

Y, U, V = raw_frame.planes
print(f"{raw_frame.pixel_format = }, {raw_frame.bit_depth = }")
print(f"{raw_frame.color_space = }, {raw_frame.color_range = }")
print(f"{Y.shape = }, {U.shape = }, {Y.dtype = }, {Y.stride() = }")

# %%
# There is one tensor per component of :attr:`RawFrame.pixel_format`, in the order
# that format describes, and only the luma and alpha ones are full size: the
# chroma planes are subsampled by whatever the format says - half in both
# directions for the 4:2:0 formats above.
#
# These are views into the frame's memory, so they are not contiguous in
# general: the row stride is the decoder's own line size, and the chroma planes
# of a semi-planar format such as an NVDEC ``nv12`` surface are two interleaved
# views over a single allocation. Writing through them is visible to whatever
# reads the :class:`RawFrame` next.
#
# Being the decoder's own planes, they are also never rotated - a video whose
# container asks for a rotation gives you the samples as they were encoded, and
# :attr:`RawFrame.rotation` tells you what to apply. A :class:`ColorConverter`
# applies it for you.

# %%
# Doing the conversion yourself
# -----------------------------
#
# With :attr:`RawFrame.planes` and the metadata beside it
# (:attr:`~RawFrame.color_space`, :attr:`~RawFrame.color_range`,
# :attr:`~RawFrame.bit_depth`), the conversion is yours to write. Here it's
# plain PyTorch ops - it could just as well be a Triton or CUDA kernel, fused
# with whatever your model needs next.

assert raw_frame.pixel_format in ("yuv420p", "nv12")  # 8-bit 4:2:0, CPU and CUDA
assert raw_frame.color_space == "bt709" and raw_frame.color_range == "tv"


def yuv420_to_rgb(Y, U, V):
    # BT.709, limited range. Chroma is upsampled by nearest neighbour.
    height, width = Y.shape

    def upsample(plane):
        plane = (plane.float() - 128) * (255 / 224)
        return plane.repeat_interleave(2, 0).repeat_interleave(2, 1)[:height, :width]

    y = (Y.float() - 16) * (255 / 219)
    u, v = upsample(U), upsample(V)
    rgb = torch.stack(
        [
            y + 1.5748 * v,
            y - 0.1873 * u - 0.4681 * v,
            y + 1.8556 * u,
        ]
    )
    return rgb.round_().clamp_(0, 255).to(torch.uint8)


ours = yuv420_to_rgb(Y, U, V)
reference = ColorConverter(device=device).convert(raw_frame).data
print(f"{ours.shape = }, mean abs diff vs ColorConverter: "
      f"{(ours.float() - reference.float()).abs().mean():.2f}")

# %%
# CUDA streams
# ^^^^^^^^^^^^
#
# On CUDA, a :class:`RawFrame` comes with one promise:
#
#    Its samples are produced on the stream that was current when you called
#    :meth:`VideoPacketDecoder.decode`, and all of that work has been enqueued
#    by the time the call returns.
#
# If you consume the frame on that same stream, you are done - stop reading,
# stream ordering handles everything. The rest of this section is for the case
# where the decoder runs on one stream and you consume the frame on another.
# It applies just as much when the consumer is a :class:`ColorConverter`: it is
# an ordinary consumer, and it does no synchronization on your behalf.
#
# Reading the samples
# """""""""""""""""""
#
# The frame's samples arrive through an asynchronous copy, so a consumer on
# another stream has to wait for it. Record an event on the decoding stream and
# wait on it::
#
#     with torch.cuda.stream(decode_stream):
#         frames = packet_decoder.decode(packet)
#         decoded = torch.cuda.Event()
#         decoded.record()
#
#     decoded.wait(my_stream)
#     with torch.cuda.stream(my_stream):
#         rgb = yuv420_to_rgb(*frames[0].planes)
#
# ``my_stream.wait_stream(decode_stream)`` works too, but it is blunter: it
# waits for *everything* queued on the decoding stream, so a consumer that has
# fallen behind a decoder running ahead of it ends up waiting for the whole
# backlog rather than for its own frame.
#
# Letting the frame go
# """"""""""""""""""""
#
# The other half is less obvious. The samples live in a PyTorch CUDA
# allocation made on the decoding stream, and the caching allocator only ever
# hands a freed block back out to an allocation on that same stream. So once
# you drop your last reference, a later frame's storage can land on top of
# these samples - while your reads are still queued. It shows up as the
# occasional corrupted frame, never as an error.
#
# You have two ways out, and they cost different things.
#
# **Sync back before dropping the frame.** Make the decoding stream wait for
# your reads, then let go::
#
#     decode_stream.wait_stream(my_stream)
#     del frames
#
# This is precise and costs no extra memory, but it stalls the decoder: its
# next frame cannot start until your reads have finished, which is exactly the
# overlap you built a second stream to get. Recover it by working in batches -
# hold a handful of frames, sync once, drop them all - so the decoder stalls
# once every N frames instead of every frame. N is then a plain memory knob.
#
# **Or hand the problem to the allocator** with
# :meth:`torch.Tensor.record_stream`, on :attr:`RawFrame.storage_cuda`::
#
#     with torch.cuda.stream(my_stream):
#         rgb = yuv420_to_rgb(*frames[0].planes)
#         frames[0].storage_cuda.record_stream(my_stream)
#
# This never stalls the decoder. Instead the allocator withholds the block
# until your reads are done, so you pay in memory rather than in latency - and
# the amount is decided by device timing rather than by you. Under memory
# pressure the allocator falls back to blocking and releasing cached blocks,
# which is a large stall at an unpredictable moment. Prefer it when you have
# headroom to spare and latency you don't.
#
# .. warning::
#
#    ``record_stream`` has to be called on :attr:`RawFrame.storage_cuda`.
#    Calling it on a plane is silently a no-op: the planes are views that the
#    allocator knows nothing about, so it ignores them and your buffer stays
#    unprotected.
#
# Whichever you pick, "dropping the frame" means dropping the *last* reference
# to its samples. The planes outlive the :class:`RawFrame` they came from - each
# one keeps the buffer alive on its own - so a stray ``Y`` still in scope is
# holding the frame's memory, and a ``del raw_frame`` on its own has released
# nothing.

# %%
# Formats that can't be viewed
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Not every pixel format can be exposed as a tensor without a copy.
# :attr:`RawFrame.planes` raises a ``RuntimeError`` for the sub-byte-packed,
# palettised and float formats, and for frames stored bottom-up. Check
# :attr:`~RawFrame.pixel_format` first if you are decoding something exotic; on CUDA
# you never have to, since NVDEC only ever produces a handful of surface
# formats (``nv12``, ``p010le``, ``p012le``, ``p016le``, ``yuv444p``,
# ``yuv444p16le``).

# %%
# Raw HDR frames
# --------------
#
# Raw planes come at the source's own precision, so a 10-bit HDR video gives
# ``uint16`` planes with all 10 bits intact - no clipping to 8 bits, and no
# tone mapping. (For the :class:`~torchcodec.decoders.VideoDecoder` route to
# HDR, through its ``output_dtype`` parameter, see
# :ref:`sphx_glr_generated_examples_decoding_hdr_decoding.py`.)
hdr_video_path = temp_dir / "hdr.mp4"
subprocess.run(
    [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=1",
        "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-preset", "ultrafast",
        "-x265-params",
        "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited",
        str(hdr_video_path),
    ],
    check=True,
    capture_output=True,  # x265 logs its banner to stderr no matter what
)

hdr_demuxer = Demuxer(hdr_video_path)
hdr_packet_decoder = hdr_demuxer.streams[0].make_decoder(device=device)
hdr_raw = next(decode_raw(hdr_demuxer, hdr_packet_decoder))

hdr_Y = hdr_raw.planes[0]
print(f"{hdr_raw.pixel_format = }, {hdr_raw.bit_depth = }, {hdr_Y.dtype = }")
print(f"{hdr_raw.color_space = }, {hdr_raw.color_primaries = }, "
      f"{hdr_raw.color_transfer_characteristic = }")

# %%
# A plane's dtype only tells you its storage width, ``uint8`` or ``uint16``;
# :attr:`RawFrame.bit_depth` is what tells you the range of the values held in
# it, and so what to shift or scale by.
#
# NVDEC surfaces are 16-bit containers holding the samples msb-aligned, so the
# 10 bits sit at the top and the low 6 are zero. Shift them back down to read
# the sample values.
shift = 16 - hdr_raw.bit_depth if device == "cuda" else 0
samples = hdr_Y.to(torch.int32) >> shift
print(f"luma range: [{samples.min()}, {samples.max()}], "
      f"{2 ** hdr_raw.bit_depth} levels available")

# %%
# Raw audio samples
# -----------------
#
# The audio side is similar. An
# :class:`AudioPacketDecoder` hands out :class:`RawAudioSamples` objects, whose
# :attr:`~RawAudioSamples.data` is always a contiguous
# ``[num_channels, num_samples]`` tensor - planar and packed sources alike - in
# whichever dtype holds the source's samples exactly.
#
# One thing differs from video: these samples are a *copy*, not a view.
# :attr:`RawFrame.planes` aliases the buffer of the frame as the decoder
# produced it, whereas :attr:`~RawAudioSamples.data` is a copy so that all
# samples come out as ``[num_channels, num_samples]``. It is still the tensor an
# :class:`AudioConverter` reads, though, so mutating it in place does change
# what the converter gives you.
audio_path = temp_dir / "audio.wav"
subprocess.run(
    [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1",
        "-c:a", "pcm_s16le", str(audio_path),
    ],
    check=True,
)

demuxer = Demuxer(audio_path, streams="audio")
packet_decoder = demuxer.streams[0].make_decoder()
raw_samples = next(decode_raw(demuxer, packet_decoder))

print(f"{raw_samples.data.dtype = }, {raw_samples.data.shape = }, "
      f"{raw_samples.sample_rate = }")
print(f"value range: [{raw_samples.data.min()}, {raw_samples.data.max()}]")

# %%
# An ``s16`` source gives ``int16``, an ``s32`` source ``int32``, an ``fltp``
# source ``float32``, and so on. The integer ones are *not* normalised to
# ``[-1, 1]``: that, along with resampling and remixing, is what an
# :class:`AudioConverter` does. If normalising is all you need, you can always
# do it manually:
normalised = raw_samples.data.to(torch.float32) / 2 ** 15
# FFmpeg's `sine` source is a quiet tone, so this doesn't reach ±1.
print(f"{normalised.dtype = }, "
      f"range [{normalised.min():.3f}, {normalised.max():.3f}]")
