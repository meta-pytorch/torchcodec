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

This tutorial is about one situation: running a :class:`VideoPacketDecoder` on
one CUDA stream and consuming its :class:`RawFrame`\\ s on another.

**If everything runs on the same CUDA stream, you can stop reading.** That is
the default, it is what you get unless you create a stream yourself, and stream
ordering takes care of everything. Running the stages on several *threads* does
not by itself put them on different streams either - see
:ref:`sphx_glr_generated_examples_blocks_pipelines.py`.

If you do reach for a second stream, read on: a :class:`RawFrame` on CUDA is a
view into a PyTorch CUDA allocation that the decoder is still responsible for,
and crossing streams with one makes you responsible for two things. Neither
produces an error when you get it wrong - both show up as the occasional
corrupted frame.

The promise
-----------

Everything below rests on one guarantee:

.. important::

   A :class:`RawFrame`'s samples are produced on the CUDA stream that was
   current when you called :meth:`VideoPacketDecoder.decode`, and all of that
   work has been enqueued by the time the call returns.

So "the decoding stream" always means a stream you chose. Note that this applies
to a :class:`ColorConverter` too: it is an ordinary consumer of a
:class:`RawFrame`, it runs on whatever stream is current when you call
:meth:`~ColorConverter.convert`, and it does no synchronization on your behalf.
Everything here applies whether you convert with a :class:`ColorConverter` or
read :attr:`RawFrame.planes` yourself.
"""

# %%
# Waiting for the samples
# -----------------------
#
# The samples reach the frame through an asynchronous copy, so a consumer on
# another stream has to wait for that copy before reading. Record an event on
# the decoding stream and wait on it::
#
#     with torch.cuda.stream(decode_stream):
#         frames = packet_decoder.decode(packet)
#         decoded = torch.cuda.Event()
#         decoded.record()
#
#     decoded.wait(consumer_stream)
#     with torch.cuda.stream(consumer_stream):
#         rgb = color_converter.convert(frames[0])
#
# ``consumer_stream.wait_stream(decode_stream)`` also works, and is shorter, but
# it is blunt: it waits for *everything* currently queued on the decoding
# stream. A consumer that has fallen behind a decoder running ahead of it ends
# up waiting for the entire backlog instead of for its own frame, which is the
# overlap you wanted in the first place.
#
# Letting the frame go
# --------------------
#
# The second obligation is the less obvious one, and it is about memory rather
# than data.
#
# A frame's samples live in a PyTorch CUDA allocation made on the decoding
# stream. The caching allocator only ever hands a freed block back out to an
# allocation on that same stream - which is what makes it safe to reuse memory
# without synchronizing, as long as everything stays on one stream. It also
# means that once you drop your last reference to a frame, a later frame's
# storage can be placed on exactly those bytes, and be written there while your
# reads are still queued on your own stream. The
# `Streams and freeing memory
# <https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html>`_ section
# of Zach DeVito's post on the allocator describes the same hazard in general.
#
# You have two ways out. They are both correct, and they cost different things.
#
# Option 1: sync back before dropping the frame
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Make the decoding stream wait for your reads, then let the frame go::
#
#     decode_stream.wait_stream(consumer_stream)
#     del frames
#
# Your memory use is exactly what your code holds, and there is no allocator
# behaviour to reason about. The cost is that it stalls the decoder: its next
# frame cannot start until your reads have finished, which removes the overlap
# you built a second stream to get.
#
# You get the overlap back by working in batches - hold a handful of frames,
# sync once, drop them all together - so the decoder stalls once every N frames
# instead of once per frame. N is then an explicit memory knob: N frames'
# worth of surfaces, chosen by you.
#
# Option 2: let the allocator handle it
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :meth:`torch.Tensor.record_stream` tells the allocator that another stream is
# using the block, so it withholds it from reuse until your reads complete.
# Call it on :attr:`RawFrame.storage_cuda`, right after queueing your reads::
#
#     with torch.cuda.stream(consumer_stream):
#         rgb = color_converter.convert(frames[0])
#         frames[0].storage_cuda.record_stream(consumer_stream)
#
# This never stalls the decoder, so you keep full overlap. You pay in memory
# instead of latency, and the amount is decided by device timing rather than by
# you: the allocator polls the recorded stream on later allocations, so peak
# memory varies from run to run and cannot be attributed to any line of your
# code. If you run close to your memory ceiling, the allocator's fallback is to
# block and release cached blocks - a large stall at an unpredictable moment.
#
# Reach for it when you have memory headroom to spare and latency you don't.
#
# .. warning::
#
#    ``record_stream`` only works on :attr:`RawFrame.storage_cuda`. Calling it
#    on a plane is silently a no-op - the planes are views the allocator knows
#    nothing about, so it ignores them and your buffer stays unprotected. There
#    is no error; you simply get the race you were trying to avoid.
#
# Whichever option you choose, "dropping the frame" means dropping the *last*
# reference to its samples. The planes outlive the :class:`RawFrame` they came
# from - each one keeps the buffer alive on its own - so a stray ``Y`` still in
# scope is holding the frame's memory, and a ``del raw_frame`` on its own may
# have released nothing.

# %%
# Putting it together
# -------------------
#
# A complete two-stream pipeline, decoding on one stream and converting on
# another, with a batched sync-back. This only runs if a GPU is available.
import torch

if not torch.cuda.is_available():
    print("No CUDA device available, skipping.")
else:
    import subprocess
    import tempfile
    from pathlib import Path

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
                packet_decoder.drain() if packet is None
                else packet_decoder.decode(packet)
            )
            decoded = torch.cuda.Event()
            decoded.record()

        decoded.wait(convert_stream)
        with torch.cuda.stream(convert_stream):
            for raw_frame in raw_frames:
                frames.append(color_converter.convert(raw_frame))

        # Hold the raw frames until the converter is done with them, then let
        # the decoder catch up with a single sync and release the batch.
        in_flight += raw_frames
        if len(in_flight) >= BATCH:
            decode_stream.wait_stream(convert_stream)
            in_flight.clear()

    decode_stream.wait_stream(convert_stream)
    in_flight.clear()
    torch.cuda.synchronize()

    print(f"{len(frames)} frames, {frames[0].data.shape = }")
