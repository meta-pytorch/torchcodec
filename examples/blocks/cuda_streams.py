# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
=======================
Blocks and CUDA streams
=======================

.. currentmodule:: torchcodec.decoders._blocks

.. warning::

   **The Blocks APIs are under active construction.** They are private
   and unreleased. Signatures and semantics may change without notice. This
   tutorial only exists to show what they will eventually make possible.

On CUDA, if you decode on one stream and consume the :class:`RawFrame`\\ s on
another - either with a :class:`ColorConverter`, or with your own consumer as in
:ref:`sphx_glr_generated_examples_blocks_raw_data.py` - you must take care of
two things:

1. Waiting for the decoder's asynchronous copy before reading the samples.
2. Keeping the CUDA caching allocator from reusing the frame's memory while
   your reads are still queued.

Getting this wrong will lead to occasional corrupted frames. We cover both
points below.

Note that if you don't create a separate stream for consuming the `RawFrame`,
none of this applies: everything runs on the same stream and stream ordering
handles it. 
"""

# %%
# Waiting for the samples to be ready
# -----------------------------------
#
# A :class:`RawFrame`'s samples are produced on the stream that was current when
# you called :meth:`VideoPacketDecoder.decode`.
# :meth:`VideoPacketDecoder.decode` typically involves a CUDA copy of the NVDEC
# decoder internals into a PyTorch-allocated CUDA tensor. This copy is
# asynchronous, so a consumer on another stream has to wait for it to finish.
# For that, you can record an event on the decoding stream and wait on it::
#
#     with torch.cuda.stream(decode_stream):
#         raw_frames = packet_decoder.decode(packet)
#         decoded = torch.cuda.Event()
#         decoded.record()
#
#     decoded.wait(convert_stream)
#     with torch.cuda.stream(convert_stream):
#         frame = color_converter.convert(raw_frames[0])
#
# Alternatively, you can use ``convert_stream.wait_stream(decode_stream)``, but
# it has different semantics: it waits for everything currently queued on the
# decoding stream, so a consumer that has fallen behind waits for the decoder's
# whole backlog instead of for its own frame.

# %%
# Letting the frame go
# --------------------
#
# The frame's samples live in a PyTorch CUDA allocation made on the decoding
# stream, and the caching allocator only ever hands a freed block back out to an
# allocation on that same stream. So once you drop your last reference, a later
# frame's storage can be placed on those same bytes and written there while your
# reads are still queued. See the `Streams and freeing memory
# <https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html>`_ section
# of Zach DeVito's post on the allocator.
#
# Note that "dropping the frame" means dropping the *last* reference to its
# samples. The planes outlive the :class:`RawFrame` they came from - each one
# keeps the buffer alive on its own - so ``del raw_frame`` while a plane is
# still in scope has released nothing.
#
# You have two ways to handle this. Both are correct, and they cost different
# things.
#
# Option 1: sync back before dropping the frame
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Make the decoding stream wait for your reads, then let the frame go::
#
#     decode_stream.wait_stream(convert_stream)
#     del raw_frames
#
# Memory use is exactly what your code holds. The cost is that the decoder's
# next frame cannot start until your reads have finished, which removes the
# overlap you created a second stream for. Work in batches to get it back: hold
# N frames, sync once, drop them together. The decoder then stalls once every N
# frames, and N is an explicit memory knob.
#
# Option 2: let the allocator handle it
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :meth:`torch.Tensor.record_stream` tells the allocator that another stream is
# using the block, so it withholds it from reuse until your reads complete. Call
# it on :attr:`RawFrame.storage_cuda`, right after queueing your reads::
#
#     with torch.cuda.stream(convert_stream):
#         frame = color_converter.convert(raw_frames[0])
#         raw_frames[0].storage_cuda.record_stream(convert_stream)
#
# This never stalls the decoder, so you keep full overlap, and you pay in memory
# instead. How much is decided by device timing rather than by you: the
# allocator polls the recorded stream on later allocations, so peak memory
# varies between runs and can't be attributed to a line of your code. Close to
# your memory ceiling, the allocator's fallback is to block and release cached
# blocks, which is a large stall at an unpredictable point.
#
# .. warning::
#
#    ``record_stream`` only works on :attr:`RawFrame.storage_cuda`. Calling it
#    on a plane is silently a no-op - the planes are views the allocator knows
#    nothing about - and leaves your buffer unprotected.

# %%
# Putting it together
# -------------------
#
# A two-stream pipeline: decoding on one stream, converting on another, with a
# batched sync-back.
import subprocess
import tempfile
from pathlib import Path

import torch

from torchcodec.decoders._blocks import ColorConverter, Demuxer

video_path = Path(tempfile.mkdtemp()) / "video.mp4"
subprocess.run(
    [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30",
        str(video_path),
    ],
    check=True,
)

BATCH = 8

demuxer = Demuxer(video_path)
packet_decoder = demuxer.streams[0].make_decoder(device="cuda")
color_converter = ColorConverter(device="cuda")

decode_stream = torch.cuda.Stream()
convert_stream = torch.cuda.Stream()

frames, in_flight = [], []
for packet in list(demuxer) + [None]:
    with torch.cuda.stream(decode_stream):
        raw_frames = (
            packet_decoder.drain() if packet is None else packet_decoder.decode(packet)
        )
        decoded = torch.cuda.Event()
        decoded.record()

    decoded.wait(convert_stream)
    with torch.cuda.stream(convert_stream):
        for raw_frame in raw_frames:
            frames.append(color_converter.convert(raw_frame))

    # Hold the raw frames until the converter is done with them, then let the
    # decoder catch up with a single sync and release the batch.
    in_flight += raw_frames
    if len(in_flight) >= BATCH:
        decode_stream.wait_stream(convert_stream)
        in_flight.clear()

decode_stream.wait_stream(convert_stream)
in_flight.clear()
torch.cuda.synchronize()

print(f"{len(frames)} frames, {frames[0].data.shape = }")
