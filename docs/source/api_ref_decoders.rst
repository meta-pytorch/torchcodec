.. _decoders:

===================
torchcodec.decoders
===================

.. currentmodule:: torchcodec.decoders



Video Decoding
--------------

For a video decoder tutorial, see: :ref:`sphx_glr_generated_examples_decoding_basic_example.py`.

The :class:`VideoDecoder` class is a high-level entry point for video decoding.
For low-level control over the decoding process, see :ref:`decoders_low_level`.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    VideoDecoder

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: dataclass.rst

    VideoStreamMetadata

**CUDA decoding utils:**

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    set_cuda_backend
    set_nvdec_cache_capacity
    get_nvdec_cache_capacity


.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: dataclass.rst

    CpuFallbackStatus

Audio Decoding
--------------

For an audio decoder tutorial, see: :ref:`sphx_glr_generated_examples_decoding_audio_decoding.py`.

The :class:`AudioDecoder` class is a high-level entry point for audio decoding.
For low-level control over the decoding process, see :ref:`decoders_low_level`.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    AudioDecoder
    WavDecoder

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: dataclass.rst

    AudioStreamMetadata


Image Decoding
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    decode_image
    decode_jpeg
    decode_png
    decode_webp
    decode_gif
    decode_avif
    decode_heic

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: enum.rst

    ImageReadMode


.. _decoders_low_level:

Low-level decoding APIs for video and audio
-------------------------------------------

.. important::

   **The low-level APIs are in beta.** Their signatures and semantics may still
   change slightly, in response to user feedback. Please `share your feedback
   <https://github.com/meta-pytorch/torchcodec/issues?q=is:open+is:issue>`__!

These APIs expose the three stages of decoding as separate objects: demuxing, decoding and conversion.
Reach for them when you need to control a stage, skip one, or run them on
different threads. If you just want RGB frames out of a file,
:class:`VideoDecoder` and :class:`AudioDecoder` are easier to use.

For tutorials, see:

- :ref:`sphx_glr_generated_examples_low_level_basics.py`
- :ref:`sphx_glr_generated_examples_low_level_pipelines.py`
- :ref:`sphx_glr_generated_examples_low_level_raw_data.py`
- :ref:`sphx_glr_generated_examples_low_level_cuda_streams.py`

Demuxing
^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    Demuxer
    VideoStream
    AudioStream

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    get_container_metadata

Decoding
^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class_inherited.rst

    VideoPacketDecoder
    AudioPacketDecoder

Conversion
^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    ColorConverter
    AudioConverter

Data types
^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    Packet
    RawFrame

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: dataclass.rst

    FrameIndex
    RawAudioSamples

Metadata
^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: dataclass.rst

    DemuxerMetadata
    ContainerMetadata
    VideoStreamHeaderMetadata
    AudioStreamHeaderMetadata
    StreamMetadata
