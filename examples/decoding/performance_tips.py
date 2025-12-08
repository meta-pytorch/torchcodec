# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
.. meta::
   :description: Learn how to optimize TorchCodec video decoding performance with batch APIs, approximate seeking, multi-threading, and CUDA acceleration.

==============================================
TorchCodec Performance Tips and Best Practices
==============================================

This tutorial consolidates performance optimization techniques for video
decoding with TorchCodec. Learn when and how to apply various strategies
to increase performance.
"""


# %%
# Overview
# --------
#
# When decoding videos with TorchCodec, several techniques can significantly
# improve performance depending on your use case. This guide covers:
#
# 1. **Batch APIs** - Decode multiple frames at once
# 2. **Approximate Mode & Keyframe Mappings** - Trade accuracy for speed
# 3. **Multi-threading** - Parallelize decoding across videos or chunks
# 4. **CUDA Acceleration** - Use GPU decoding for supported formats
#
# We'll explore each technique and when to use it.

# %%
# 1. Use Batch APIs When Possible
# --------------------------------
#
# If you need to decode multiple frames at once, the batch methods are faster than calling single-frame decoding methods multiple times.
# For example, :meth:`~torchcodec.decoders.VideoDecoder.get_frames_at` is faster than calling :meth:`~torchcodec.decoders.VideoDecoder.get_frame_at` multiple times.
# TorchCodec's batch APIs reduce overhead and can leverage internal optimizations.
#
# **Key Methods:**
#
# For index-based frame retrieval:
#
# - :meth:`~torchcodec.decoders.VideoDecoder.get_frames_at` for specific indices
# - :meth:`~torchcodec.decoders.VideoDecoder.get_frames_in_range` for ranges
#
# For timestamp-based frame retrieval:
#
# - :meth:`~torchcodec.decoders.VideoDecoder.get_frames_played_at` for timestamps
# - :meth:`~torchcodec.decoders.VideoDecoder.get_frames_played_in_range` for time ranges
#
# **When to use:**
#
# - Decoding multiple frames

# %%
# .. note::
#
#     For complete examples with runnable code demonstrating batch decoding,
#     iteration, and frame retrieval, see:
#
#     - :ref:`sphx_glr_generated_examples_decoding_basic_example.py`

# %%
# 2. Approximate Mode & Keyframe Mappings
# ----------------------------------------
#
# By default, TorchCodec uses ``seek_mode="exact"``, which performs a :term:`scan` when
# you create the decoder to build an accurate internal index of frames. This
# ensures frame-accurate seeking but takes longer for decoder initialization,
# especially on long videos.

# %%
# **Approximate Mode**
# ~~~~~~~~~~~~~~~~~~~~
#
# Setting ``seek_mode="approximate"`` skips the initial :term:`scan` and relies on the
# video file's metadata headers. This dramatically speeds up
# :class:`~torchcodec.decoders.VideoDecoder` creation, particularly for long
# videos, but may result in slightly less accurate seeking in some cases.
#
#
# **Which mode should you use:**
#
# - If you care about exactness of frame seeking, use “exact”.
# - If the video is long and you're only decoding a small amount of frames, approximate mode should be faster.

# %%
# **Custom Frame Mappings**
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For advanced use cases, you can pre-compute a custom mapping between desired
# frame indices and actual keyframe locations. This allows you to speed up :class:`~torchcodec.decoders.VideoDecoder`
# instantiation while maintaining the frame seeking accuracy of ``seek_mode="exact"``
#
# **When to use:**
#
# - Frame accuracy is critical, so you cannot use approximate mode
# - You can preprocess videos once and then decode them many times
#
# **Performance impact:** Enables consistent, predictable performance for repeated
# random access without the overhead of exact mode's scanning.

# %%
# .. note::
#
#     For complete benchmarks showing actual speedup numbers, accuracy comparisons,
#     and implementation examples, see:
#
#     - :ref:`sphx_glr_generated_examples_decoding_approximate_mode.py`
#
#     - :ref:`sphx_glr_generated_examples_decoding_custom_frame_mappings.py`

# %%
# 3. Multi-threading for Parallel Decoding
# -----------------------------------------
#
# When decoding multiple videos or decoding a large number of frames from a single video, there are a few parallelization strategies to speed up the decoding process:
#
# - **FFmpeg-based parallelism** - Using FFmpeg's internal threading capabilities for intra-frame parallelism, where parallelization happens within individual frames rather than across frames
# - **Multiprocessing** - Distributing work across multiple processes
# - **Multithreading** - Using multiple threads within a single process
#
# You can use both multiprocessing and multithreading to decode multiple videos in parallel, or to decode a single long video in parallel by splitting it into chunks.

# %%
# .. note::
#
#     For complete examples comparing
#     sequential, ffmpeg-based parallelism, multi-process, and multi-threaded approaches, see:
#
#     - :ref:`sphx_glr_generated_examples_decoding_parallel_decoding.py`

# %%
# 4. CUDA Acceleration
# --------------------
#
# TorchCodec supports GPU-accelerated decoding using NVIDIA's hardware decoder
# (NVDEC) on supported hardware. This keeps decoded tensors in GPU memory,
# avoiding expensive CPU-GPU transfers for downstream GPU operations.
#
# **When to use:**
#
# - Decoding large resolution videos
# - Large batch of videos saturating the CPU
# - GPU-intensive pipelines with transforms like scaling and cropping
# - CPU is saturated and you want to free it up for other work
#
# **When NOT to use:**
#
# - You need bit-exact results
# - Small resolution videos and the PCI-e transfer latency is large
# - GPU is already busy and CPU is idle
#
# **Performance impact:** CUDA decoding can significantly outperform CPU decoding,
# especially for high-resolution videos and when combined with GPU-based transforms.
# Actual speedup varies by hardware, resolution, and codec.

# %%
# **Recommended Usage for Beta Interface**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. code-block:: python
#
#     with set_cuda_backend("beta"):
#         decoder = VideoDecoder("file.mp4", device="cuda")
#

# %%
# **Checking for CPU Fallback**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In some cases, CUDA decoding may silently fall back to CPU decoding when the
# video codec or format is not supported by NVDEC. You can detect this using
# the :attr:`~torchcodec.decoders.VideoDecoder.cpu_fallback` attribute:
#
# .. code-block:: python
#
#     decoder = VideoDecoder("file.mp4", device="cuda")
#     decoder[0]  # Decode at least one frame first (for FFmpeg backend)
#
#     # Print detailed fallback status
#     print(decoder.cpu_fallback)
#
# .. note::
#
#     The timing of when you can detect CPU fallback differs between backends:
#
#     - **FFmpeg backend**: You can only check fallback status after decoding at
#       least one frame, because FFmpeg determines codec support lazily during decoding.
#     - **BETA backend**: You can check fallback status immediately after
#       decoder creation, as the backend checks codec support upfront.
#
#     For installation instructions, detailed examples, and visual comparisons
#     between CPU and CUDA decoding, see:
#
#     - :ref:`sphx_glr_generated_examples_decoding_basic_cuda_example.py`

# %%
# Conclusion
# ----------
#
# TorchCodec offers multiple performance optimization strategies, each suited to
# different scenarios. Use batch APIs for multi-frame decoding, approximate mode
# for faster initialization, parallel processing for high throughput, and CUDA
# acceleration for GPU-intensive workflows.
#
# The best results often come from combining techniques. Profile your specific
# use case and apply optimizations incrementally, using the benchmarks in the
# linked examples as a guide.
#
# For more information, see:
#
# - :ref:`sphx_glr_generated_examples_decoding_basic_example.py` - Basic decoding examples
# - :ref:`sphx_glr_generated_examples_decoding_approximate_mode.py` - Approximate mode benchmarks
# - :ref:`sphx_glr_generated_examples_decoding_custom_frame_mappings.py` - Custom frame mappings
# - :ref:`sphx_glr_generated_examples_decoding_parallel_decoding.py` - Parallel decoding strategies
# - :ref:`sphx_glr_generated_examples_decoding_basic_cuda_example.py` - CUDA acceleration guide
# - :class:`torchcodec.decoders.VideoDecoder` - Full API reference
